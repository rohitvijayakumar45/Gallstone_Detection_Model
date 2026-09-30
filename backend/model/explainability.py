"""
explainability.py
-----------------
Three visualisations, all derived from the same set of perturbed inference
passes so we only run the model N times total (not 3×N).

  1. Detection Consistency Map  — mean detection heat across N perturbed passes.
     Bright regions = model consistently fires there regardless of noise.
     More faithful to actual detections than backbone-PCA (EigenCAM).

  2. Segmentation Mask          — primary YOLO26-seg masks + yellow outlines for
     any region stability found but the primary pass missed.

  3. Prediction Stability       — variance of detection heat across passes.
     High variance (bright) → model is uncertain in that region.

All three return a base64-encoded PNG data URL.
Call compute_all_explainability() to get all three in one shot.
"""

import io
import os
import base64
import logging
import tempfile

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# ── Colour palette for primary-detection instances ───────────────────────────
_INSTANCE_COLOURS = [
    (220,  38,  38),   # red
    ( 37,  99, 235),   # blue
    (  5, 150, 105),   # emerald
    (217, 119,   6),   # amber
    (139,  92, 246),   # violet
    ( 14, 165, 233),   # sky
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _np_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _apply_colormap(gray: np.ndarray, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Apply an OpenCV colormap to a [0,1] float32 array → RGB uint8."""
    norm = (gray * 255).clip(0, 255).astype(np.uint8)
    bgr  = cv2.applyColorMap(norm, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _smooth_normalise(heat: np.ndarray, H: int, W: int) -> np.ndarray:
    smoothed = gaussian_filter(heat, sigma=max(H, W) / 80)
    m = smoothed.max()
    if m > 0:
        smoothed /= m
    return smoothed


# ── Shared inference passes ───────────────────────────────────────────────────

def _run_passes(yolo_model, image_path: str, n_passes: int):
    """
    Run N perturbed inference passes.

    Returns
    -------
    img_rgb    : (H, W, 3) uint8
    heat_maps  : list of (H, W) float32 — summed detection confidence per pixel
    pass_confs : list of float — max detection confidence per pass (0 if none)
    """
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    H, W    = img_rgb.shape[:2]
    rng     = np.random.default_rng(seed=42)
    heat_maps  = []
    pass_confs = []

    for i in range(n_passes):
        noise      = rng.normal(0, 5 + i, img_rgb.shape).astype(np.float32)
        brightness = 1.0 + rng.uniform(-0.10, 0.10)
        blur_k     = max(1, int(rng.uniform(0, 2)) * 2 + 1)

        perturbed = (img_rgb.astype(np.float32) * brightness + noise).clip(0, 255).astype(np.uint8)
        if blur_k > 1:
            perturbed = cv2.GaussianBlur(perturbed, (blur_k, blur_k), 0)

        buf = io.BytesIO()
        Image.fromarray(perturbed).save(buf, format="JPEG", quality=95)
        buf.seek(0)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name

        try:
            results = yolo_model.predict(source=tmp_path, conf=0.15, save=False, verbose=False)
            result  = results[0]
        finally:
            os.unlink(tmp_path)

        heat     = np.zeros((H, W), dtype=np.float32)
        max_conf = 0.0
        for box in result.boxes:
            conf = float(box.conf[0])
            max_conf = max(max_conf, conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            heat[y1:y2, x1:x2] += conf

        heat_maps.append(heat)
        pass_confs.append(max_conf)

    return img_rgb, heat_maps, pass_confs


# ── Per-visualisation renderers ───────────────────────────────────────────────

def _render_consistency(img_rgb, mean_smooth):
    """JET overlay of mean detection heat (Detection Consistency Map)."""
    H, W    = img_rgb.shape[:2]
    colored = _apply_colormap(mean_smooth, cv2.COLORMAP_JET)
    blended = (0.5 * colored + 0.5 * img_rgb).clip(0, 255).astype(np.uint8)
    return _np_to_b64(blended)


def _render_segmask(result_obj, pil_image, mean_smooth, missed_threshold=0.30):
    """
    Primary seg masks + yellow dashed outlines for stability-detected missed regions.

    A region counts as "missed" when:
      • its mean stability heat exceeds `missed_threshold`, AND
      • it has no overlap with any primary-inference mask/box.
    """
    img  = np.array(pil_image.convert("RGB"), dtype=np.float32)
    H, W = img.shape[:2]

    primary_coverage = np.zeros((H, W), dtype=np.uint8)

    # ── Draw primary masks ────────────────────────────────────────────────────
    if result_obj.masks is not None:
        for i, mask_data in enumerate(result_obj.masks.data):
            colour  = np.array(_INSTANCE_COLOURS[i % len(_INSTANCE_COLOURS)], dtype=np.float32)
            mask_np = mask_data.cpu().numpy().astype(np.float32)
            mask_np = cv2.resize(mask_np, (W, H))
            binary  = (mask_np > 0.5).astype(np.uint8)
            primary_coverage = np.maximum(primary_coverage, binary)

            for c in range(3):
                img[:, :, c] = np.where(
                    binary,
                    img[:, :, c] * 0.55 + colour[c] * 0.45,
                    img[:, :, c],
                )
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img.astype(np.uint8), contours, -1,
                             tuple(int(c) for c in colour), 2)

    elif len(result_obj.boxes) > 0:
        out = img.astype(np.uint8).copy()
        for i, box in enumerate(result_obj.boxes):
            colour = _INSTANCE_COLOURS[i % len(_INSTANCE_COLOURS)]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            primary_coverage[y1:y2, x1:x2] = 1
            overlay = out.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        img = out.astype(np.float32)

    # ── Overlay stability-detected missed regions ─────────────────────────────
    missed = ((mean_smooth > missed_threshold) & (primary_coverage == 0)).astype(np.uint8)
    if missed.any():
        kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        missed_closed = cv2.morphologyEx(missed, cv2.MORPH_CLOSE, kernel)
        contours, _   = cv2.findContours(missed_closed, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        out_uint = img.astype(np.uint8)
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:      # skip tiny noise blobs
                continue
            # Semi-transparent yellow fill
            mask_single = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask_single, [cnt], -1, 1, -1)
            for c_idx, val in enumerate((255, 220, 0)):
                out_uint[:, :, c_idx] = np.where(
                    mask_single,
                    (out_uint[:, :, c_idx] * 0.65 + val * 0.35).clip(0, 255).astype(np.uint8),
                    out_uint[:, :, c_idx],
                )
            # Bold yellow outline
            cv2.drawContours(out_uint, [cnt], -1, (255, 220, 0), 2)
            # "?" label at centroid
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(out_uint, '?', (cx - 6, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2, cv2.LINE_AA)
        img = out_uint.astype(np.float32)

    return _np_to_b64(img.astype(np.uint8))


def _render_stability(img_rgb, variance):
    """Plasma overlay of detection variance (original stability map)."""
    colored = _apply_colormap(variance, cv2.COLORMAP_PLASMA)
    blended = (0.55 * colored + 0.45 * img_rgb).clip(0, 255).astype(np.uint8)
    return _np_to_b64(blended)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_all_explainability(
    yolo_model,
    image_path: str,
    result_obj,
    pil_image: Image.Image,
    n_passes: int = 10,
):
    """
    Run N perturbed inference passes once and return all three visualisations
    plus a cross-perturbation confidence estimate.

    Returns
    -------
    eigencam_url      : base64 PNG — detection consistency map
    segmask_url       : base64 PNG — primary masks + stability-detected misses
    stability_url     : base64 PNG — prediction variance map
    stability_conf    : float [0,1] — 0.7×detection_rate + 0.3×mean_pass_conf.
                        Weights detection consistency heavily because per-pass
                        softmax scores are underconfident on small datasets.
    """
    img_rgb, heat_maps, pass_confs = _run_passes(yolo_model, image_path, n_passes)
    H, W = img_rgb.shape[:2]

    stack      = np.stack(heat_maps, axis=0)        # (N, H, W)
    mean_heat  = np.mean(stack, axis=0)              # consistent detection regions
    variance   = np.var(stack,  axis=0)              # uncertainty regions

    mean_smooth = _smooth_normalise(mean_heat, H, W)
    var_smooth  = _smooth_normalise(variance,  H, W)

    eigencam_url  = _render_consistency(img_rgb, mean_smooth)
    segmask_url   = _render_segmask(result_obj, pil_image, mean_smooth)
    stability_url = _render_stability(img_rgb, var_smooth)

    # ── Cross-perturbation confidence ─────────────────────────────────────────
    # The model's per-pass softmax scores are systematically underconfident
    # (a known training artefact on small datasets). The more reliable signal
    # is detection_rate — how often the model fires on this region across 10
    # independent noisy passes.
    #
    # Formula: 70% weight on detection_rate + 30% on mean per-pass confidence.
    # Example: rate=90%, mean_conf=27% → 0.9×0.7 + 0.27×0.3 = 0.63+0.08 = 71%
    detected_confs = [c for c in pass_confs if c > 0]
    detection_rate = len(detected_confs) / n_passes
    mean_pass_conf = float(np.mean(detected_confs)) if detected_confs else 0.0
    stability_conf = round(0.70 * detection_rate + 0.30 * mean_pass_conf, 4)

    logger.debug(
        "Stability confidence: rate=%.0f%% mean_conf=%.1f%% → %.1f%%",
        detection_rate * 100, mean_pass_conf * 100, stability_conf * 100,
    )

    return eigencam_url, segmask_url, stability_url, stability_conf


# ── Legacy single-call wrappers (kept for compatibility) ─────────────────────

def compute_eigencam(yolo_model, image_path: str) -> str:
    img_rgb, heat_maps, _ = _run_passes(yolo_model, image_path, n_passes=10)
    H, W = img_rgb.shape[:2]
    mean_smooth = _smooth_normalise(np.mean(np.stack(heat_maps), axis=0), H, W)
    return _render_consistency(img_rgb, mean_smooth)


def compute_segmask(result_obj, original_image: Image.Image) -> str:
    img_rgb = np.array(original_image.convert("RGB"))
    H, W    = img_rgb.shape[:2]
    blank   = np.zeros((H, W), dtype=np.float32)
    return _render_segmask(result_obj, original_image, blank)


def compute_stability(yolo_model, image_path: str, n_passes: int = 10) -> str:
    img_rgb, heat_maps, _ = _run_passes(yolo_model, image_path, n_passes)
    H, W = img_rgb.shape[:2]
    var_smooth = _smooth_normalise(np.var(np.stack(heat_maps), axis=0), H, W)
    return _render_stability(img_rgb, var_smooth)
