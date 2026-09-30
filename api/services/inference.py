"""Model loading + dual-detector inference service.

Loads YOLO + RF-DETR once at startup, exposes:

  - inference.predict(image_bgr, conf_threshold, apply_shadow=True)
      -> {boxes_pix, scores_raw, scores_after_shadow, per_model_boxes,
          per_model_scores, latencies_ms}

  - inference.calibrated_conf(scores_raw, model_name)  (uses TemperatureScaler)
  - inference.fusion_isotonic(scores)                  (uses IsotonicFusion)
  - inference.conformal_expand(boxes_norm)             (SplitConformalBox)
  - inference.crc_filter(scores)                       (CRC lambda*)

All artefacts are optional; the service degrades gracefully if calibration
or conformal files are missing (returns None for those fields).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import numpy as np


@dataclass
class InferenceState:
    yolo: Any = None
    rfdetr: Any = None
    ensemble: Any = None
    shadow: Any = None
    calibration_yolo: Any = None
    calibration_rfdetr: Any = None
    isotonic_fusion: Any = None
    conformal_qhat: Any = None
    crc: Any = None
    selective: Any = None
    device: str = "cpu"


_state = InferenceState()
_lock = Lock()


def _load_yolo(path: Path):
    if not path.exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(path))
    except Exception as exc:
        print(f"[inference] YOLO load failed: {exc}")
        return None


def _load_rfdetr(path: Path):
    if not path.exists():
        return None
    try:
        from rfdetr import RFDETRLarge
        m = RFDETRLarge(pretrain_weights=str(path))
        try:
            m.optimize_for_inference()
        except Exception:
            pass
        return m
    except Exception as exc:
        print(f"[inference] RF-DETR load failed: {exc}")
        return None


def initialise(settings) -> InferenceState:
    """Load all artefacts (idempotent, thread-safe)."""
    with _lock:
        if _state.yolo is not None and _state.rfdetr is not None:
            return _state

        _state.device = settings.resolve_device()
        _state.yolo = _load_yolo(settings.yolo_weights)
        _state.rfdetr = _load_rfdetr(settings.rfdetr_weights)

        # Ensemble
        models = [m for m in (_state.yolo, _state.rfdetr) if m is not None]
        if models:
            from src.models.ensemble import GallstoneEnsemble
            _state.ensemble = GallstoneEnsemble(models, weights=list(settings.ensemble_weights)[: len(models)])

        # Shadow prior
        try:
            from src.evaluation.shadow_analyzer import ShadowAnalyzer
            _state.shadow = ShadowAnalyzer(threshold=settings.shadow_threshold)
        except Exception as exc:
            print(f"[inference] shadow analyzer disabled: {exc}")

        # Calibration
        try:
            from src.calibration import IsotonicFusion, TemperatureScaler, SelectiveThresholds
            if settings.calibration_yolo.exists():
                _state.calibration_yolo = TemperatureScaler.load(settings.calibration_yolo)
            if settings.calibration_rfdetr.exists():
                _state.calibration_rfdetr = TemperatureScaler.load(settings.calibration_rfdetr)
            if settings.isotonic_fusion.exists():
                _state.isotonic_fusion = IsotonicFusion.load(settings.isotonic_fusion)
            if settings.selective_thresholds.exists():
                _state.selective = SelectiveThresholds.load(settings.selective_thresholds)
        except Exception as exc:
            print(f"[inference] calibration artefacts skipped: {exc}")

        # Conformal
        try:
            from src.conformal import ConformalRiskController, SplitConformalBox
            if settings.conformal_qhat.exists():
                _state.conformal_qhat = SplitConformalBox.load(settings.conformal_qhat)
            if settings.conformal_crc.exists():
                _state.crc = ConformalRiskController.load(settings.conformal_crc)
        except Exception as exc:
            print(f"[inference] conformal artefacts skipped: {exc}")

    return _state


def state() -> InferenceState:
    return _state


def predict(image_bgr: np.ndarray, conf_threshold: float,
            iou_threshold: float = 0.5,
            apply_shadow: bool = True,
            mc_samples: int = 0) -> dict:
    if _state.ensemble is None:
        raise RuntimeError("no models loaded")
    h, w = image_bgr.shape[:2]

    # Per-model counts for UI provenance.
    per_model: dict[str, int] = {"yolo": 0, "rfdetr": 0}
    per_model_latency_ms: dict[str, float] = {}
    if _state.yolo is not None:
        try:
            t0 = time.perf_counter()
            yres = _state.yolo.predict(source=image_bgr, conf=conf_threshold, verbose=False)[0]
            per_model_latency_ms["yolo"] = round((time.perf_counter() - t0) * 1000.0, 2)
            per_model["yolo"] = int(yres.boxes.shape[0]) if yres.boxes is not None else 0
        except Exception:
            pass
    if _state.rfdetr is not None:
        try:
            import cv2 as _cv2
            rgb = _cv2.cvtColor(image_bgr, _cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            rres = _state.rfdetr.predict(rgb, threshold=conf_threshold)
            per_model_latency_ms["rfdetr"] = round((time.perf_counter() - t0) * 1000.0, 2)
            if hasattr(rres, "xyxy"):
                per_model["rfdetr"] = int(len(rres.xyxy))
            elif isinstance(rres, tuple) and len(rres) == 3:
                per_model["rfdetr"] = int(len(rres[0]))
        except Exception:
            pass

    t0 = time.perf_counter()
    boxes, scores, labels = _state.ensemble.predict(
        image_bgr, conf_threshold=conf_threshold, iou_threshold=iou_threshold
    )
    fusion_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[DEBUG predict] img_shape={image_bgr.shape} conf_threshold={conf_threshold} "
          f"per_model={per_model} fused_n={len(scores)} fused_scores={np.asarray(scores).tolist()}")

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size and boxes.max() <= 1.05:
        boxes_pix = boxes * np.array([w, h, w, h], dtype=np.float32)
    else:
        boxes_pix = boxes

    scores_after_shadow = scores.copy()
    shadow_ms = 0.0
    if apply_shadow and _state.shadow is not None and boxes_pix.size:
        t0 = time.perf_counter()
        scores_after_shadow = _state.shadow.apply_to_ensemble(image_bgr, boxes_pix, scores)
        shadow_ms = (time.perf_counter() - t0) * 1000.0

    aleatoric = None
    epistemic = None
    uncertainty_ms = 0.0
    if mc_samples > 0 and boxes_pix.size and _state.ensemble is not None:
        try:
            from src.explainability.uncertainty import UncertaintyEstimator

            def _pf(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                bs, sc, _lb = _state.ensemble.predict(
                    img, conf_threshold=max(0.05, conf_threshold * 0.5),
                    iou_threshold=iou_threshold
                )
                bs = np.asarray(bs, dtype=np.float32)
                sc = np.asarray(sc, dtype=np.float32)
                if bs.size and bs.max() <= 1.05:
                    bs = bs * np.array([w, h, w, h], dtype=np.float32)
                return bs, sc

            t0 = time.perf_counter()
            est = UncertaintyEstimator(n_samples=int(mc_samples))
            res = est.estimate(image_bgr, boxes_pix, _pf)
            aleatoric = res.aleatoric
            epistemic = res.epistemic
            uncertainty_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            print(f"[inference] MC uncertainty skipped: {exc}")

    return {
        "boxes_pix": boxes_pix,
        "scores_raw": scores,
        "scores_after_shadow": np.asarray(scores_after_shadow, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "per_model_counts": per_model,
        "per_model_latency_ms": per_model_latency_ms,
        "latency_ms": {
            "fusion": round(fusion_ms, 2),
            "shadow": round(shadow_ms, 2),
            "uncertainty": round(uncertainty_ms, 2),
            **{f"model_{k}": v for k, v in per_model_latency_ms.items()},
        },
        "wh": (w, h),
    }


def calibrate_fusion(scores_after_shadow: np.ndarray) -> Optional[np.ndarray]:
    if _state.isotonic_fusion is None or scores_after_shadow.size == 0:
        return None
    return np.asarray(_state.isotonic_fusion.apply(scores_after_shadow),
                      dtype=np.float32)


def conformal_boxes(boxes_pix: np.ndarray, wh: tuple[int, int]) -> Optional[np.ndarray]:
    if _state.conformal_qhat is None or boxes_pix.size == 0:
        return None
    return _state.conformal_qhat.apply_pixel(boxes_pix, wh)


def crc_keep_mask(scores: np.ndarray) -> Optional[np.ndarray]:
    if _state.crc is None or scores.size == 0:
        return None
    return _state.crc.apply(scores)


def verdict_for(prob: float, coverage_target: float = 0.9) -> str:
    """Selective-prediction verdict."""
    if _state.selective is None:
        return "detected" if prob >= 0.5 else "clear"
    tau = _state.selective.tau_at_coverage(coverage_target)
    if prob >= tau:
        return "detected"
    if prob >= tau * 0.5:
        return "review"
    return "clear"
