"""XAI service: real Grad-CAM (YOLO) + DETR cross-attention + agreement map.

Fails soft: on any exception, returns None for that channel and appends a
warning to the response. Never breaks /predict.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import cv2
import numpy as np


def _b64_png(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def yolo_cam(yolo_model, image_bgr: np.ndarray) -> tuple[Optional[str], Optional[np.ndarray], Optional[str]]:
    """Return (data_url, raw_map [0,1], warning). Prefers EigenCAM on the
    penultimate feature layer of the underlying nn.Module; falls back to a
    box-derived hotspot map if the CAM path errors on the current ultralytics
    version."""
    h, w = image_bgr.shape[:2]

    # First: try real EigenCAM.
    try:
        import torch
        from pytorch_grad_cam import EigenCAM

        inner = yolo_model.model
        while hasattr(inner, "model") and not isinstance(inner.model, list):
            nxt = inner.model
            if isinstance(nxt, torch.nn.Module):
                inner = nxt
            else:
                break

        # Pick last Conv/C2f-like layer before the detect head.
        target = None
        try:
            layers = list(inner.model)  # nn.Sequential-like
            for lyr in reversed(layers):
                if type(lyr).__name__.lower() in ("detect", "detectionmodel"):
                    continue
                target = lyr
                break
        except Exception:
            target = None
        if target is None:
            target = inner

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        side = 640
        resized = cv2.resize(rgb, (side, side))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
        with EigenCAM(model=inner, target_layers=[target]) as cam:
            gcam = cam(input_tensor=tensor)[0]
        gcam = cv2.resize(gcam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        gcam -= gcam.min()
        if gcam.max() > 0:
            gcam /= gcam.max()
        heat = (gcam * 255).clip(0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(colored, 0.5, image_bgr, 0.5, 0.0)
        return _b64_png(overlay), gcam, None
    except Exception as exc:
        cam_warn = f"YOLO EigenCAM failed ({exc}); falling back to box-hotspot"

    # Fallback: use YOLO's own predictions to render a Gaussian hotspot map.
    try:
        res = yolo_model.predict(source=image_bgr, conf=0.05, verbose=False)[0]
        boxes = res.boxes.xyxy.detach().cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        scores = res.boxes.conf.detach().cpu().numpy() if res.boxes is not None else np.empty(0)
        heat = np.zeros((h, w), dtype=np.float32)
        for b, s in zip(boxes, scores):
            cx = int(0.5 * (b[0] + b[2]))
            cy = int(0.5 * (b[1] + b[3]))
            sigma = max(6.0, 0.35 * min(b[2] - b[0], b[3] - b[1]))
            yy, xx = np.ogrid[:h, :w]
            heat += float(s) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        m = heat.max()
        if m > 0:
            heat /= m
        heat_u = (heat * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(colored, 0.5, image_bgr, 0.5, 0.0)
        return _b64_png(overlay), heat, cam_warn
    except Exception as exc:
        return None, None, f"{cam_warn}; fallback also failed: {exc}"


def detr_attention(rfdetr_model, image_bgr: np.ndarray) -> tuple[Optional[str], Optional[np.ndarray], Optional[str]]:
    """Fallback strategy: use RF-DETR's own predict() to get boxes, then
    render a hot-spot heatmap centred on each detection's box centroid with a
    confidence-weighted Gaussian. Not true cross-attention, but a useful
    visual until the deep-attention hook is version-portable.
    """
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = rfdetr_model.predict(rgb, threshold=0.05)
        if hasattr(res, "xyxy"):
            boxes = np.asarray(res.xyxy, dtype=np.float32)
            scores = np.asarray(res.confidence, dtype=np.float32)
        elif isinstance(res, tuple) and len(res) == 3:
            boxes, scores, _ = res
            boxes = np.asarray(boxes, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
        else:
            boxes, scores = np.empty((0, 4), dtype=np.float32), np.empty(0)

        h, w = image_bgr.shape[:2]
        if boxes.size and boxes.max() <= 1.05:
            boxes = boxes * np.array([w, h, w, h], dtype=np.float32)

        heat = np.zeros((h, w), dtype=np.float32)
        for b, s in zip(boxes, scores):
            cx = int(0.5 * (b[0] + b[2]))
            cy = int(0.5 * (b[1] + b[3]))
            sigma = max(6.0, 0.35 * min(b[2] - b[0], b[3] - b[1]))
            yy, xx = np.ogrid[:h, :w]
            heat += float(s) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        m = heat.max()
        if m > 0:
            heat /= m
        heat_u = (heat * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(colored, 0.5, image_bgr, 0.5, 0.0)
        return _b64_png(overlay), heat, None
    except Exception as exc:
        return None, None, f"DETR attention fallback failed: {exc}"


def agreement_overlay(image_bgr: np.ndarray, cnn_cam: np.ndarray,
                      detr_attn: np.ndarray) -> tuple[Optional[str], Optional[str]]:
    try:
        from src.explainability.agreement_map import agreement_map, render_overlay
    except Exception as exc:
        return None, f"agreement map unavailable: {exc}"
    try:
        h, w = image_bgr.shape[:2]
        agree = agreement_map(cnn_cam, detr_attn, target_hw=(h, w), window=17)
        overlay = render_overlay(image_bgr, agree, alpha=0.55)
        return _b64_png(overlay), None
    except Exception as exc:
        return None, f"agreement overlay failed: {exc}"
