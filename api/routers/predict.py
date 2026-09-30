"""Unified predict router.

Pipeline per scan:
  1. Fetch upload session (image_bgr + dicom_meta).
  2. Ensemble.predict (WBF).
  3. ShadowAnalyzer.apply_to_ensemble.
  4. Per-model temperature scaling (if artefacts loaded).
  5. Post-fusion isotonic on shadow-adjusted scores.
  6. Split-conformal box expansion + optional CRC score filter.
  7. Selective verdict from calibrated probability.
  8. Persist result in session + return.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from api.services import inference
from api.services.session import get_session, update_session
from api.settings import settings

router = APIRouter(prefix="/predict", tags=["predict"])


def _bbox(box: np.ndarray) -> dict[str, float]:
    return {
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
    }


@router.post("/{scan_id}")
async def predict(scan_id: str, conf_threshold: float | None = None,
                  iou_threshold: float | None = None,
                  mc: int = 0,
                  shadow: int = 0,
                  crc: int = 0) -> dict:
    """Run the dual-detector pipeline on a previously uploaded scan.

    Query params:
      - conf_threshold: WBF skip threshold (default 0.15 — lower than the
        historical 0.25 to preserve recall on small stones).
      - iou_threshold:  WBF IoU (default 0.50).
      - mc:             Monte-Carlo passes for aleatoric uncertainty
                        (0 disables; typical 10-20).
      - shadow:         apply posterior-acoustic-shadow prior (0 = off,
                        default: OFF because on this dataset it regresses
                        F1 vs plain WBF; see docs/BENCHMARKS.md).
      - crc:            drop detections below the CRC score threshold λ*
                        (0 = off, default: OFF so the UI shows every
                        candidate; recall guarantee still available via the
                        conformal box and calibrated probability).
    """
    sess = get_session(scan_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="scan not found")

    conf = conf_threshold if conf_threshold is not None else 0.15
    iou = iou_threshold if iou_threshold is not None else settings.iou_threshold

    try:
        res = inference.predict(sess["image_bgr"], conf_threshold=conf,
                                iou_threshold=iou,
                                apply_shadow=bool(shadow),
                                mc_samples=mc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    boxes = res["boxes_pix"]
    scores_shadow = res["scores_after_shadow"]

    # Post-fusion calibration
    scores_cal = inference.calibrate_fusion(scores_shadow)

    # Conformal
    wh = res["wh"]
    conformal = inference.conformal_boxes(boxes, wh)
    crc_mask = (inference.crc_keep_mask(scores_shadow if scores_cal is None else scores_cal)
                if crc else None)

    st = inference.state()
    alpha = st.conformal_qhat.alpha if st.conformal_qhat is not None else None

    aleatoric_arr = res.get("aleatoric")
    epistemic_arr = res.get("epistemic")

    detections: list[dict[str, Any]] = []
    for i, box in enumerate(boxes):
        raw = float(scores_shadow[i])
        cal = float(scores_cal[i]) if scores_cal is not None else None
        pass_crc = bool(crc_mask[i]) if crc_mask is not None else True
        if not pass_crc:
            continue
        verdict = inference.verdict_for(cal if cal is not None else raw)
        det = {
            "bbox": _bbox(box),
            "confidence": raw,
            "calibrated_confidence": cal,
            "shadow_score": None,  # ShadowAnalyzer folds directly into scores; keep field for schema parity
            "conformal_bbox": _bbox(conformal[i]) if conformal is not None else None,
            "conformal_alpha": alpha,
            "aleatoric": float(aleatoric_arr[i]) if aleatoric_arr is not None else None,
            "epistemic": float(epistemic_arr[i]) if epistemic_arr is not None else None,
            "verdict": verdict,
            "provenance": {
                "raw_source": "wbf+shadow",
                "calibration": "isotonic_fusion" if scores_cal is not None else None,
                "conformal": "split_conformal_bonferroni" if conformal is not None else None,
                "crc_filter": "recall_bound" if crc_mask is not None else None,
                "uncertainty": "mc_perturbation" if aleatoric_arr is not None else None,
            },
        }
        detections.append(det)

    result_payload = {
        "scan_id": scan_id,
        "detections": detections,
        "per_model_counts": res.get("per_model_counts") or {},
        "fused_count": len(detections),
        "calibration_version": "v1" if scores_cal is not None else None,
        "conformal_alpha": alpha,
        "latency_ms": res["latency_ms"],
        "disclaimer": (
            "Decision-support output. Findings must be confirmed by a clinician."
        ),
    }
    update_session(scan_id, {"prediction": result_payload})
    return result_payload


@router.get("/{scan_id}")
async def get_prediction(scan_id: str) -> dict:
    sess = get_session(scan_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="scan not found")
    if "prediction" not in sess:
        raise HTTPException(status_code=404, detail="not yet predicted")
    return sess["prediction"]
