"""XAI router. Real Grad-CAM (YOLO) + DETR cross-attention + agreement map."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.services import inference, xai
from api.services.session import get_session, update_session

router = APIRouter(prefix="/explain", tags=["explain"])


@router.get("/{scan_id}")
async def explain(scan_id: str) -> dict:
    sess = get_session(scan_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="scan not found")

    image = sess["image_bgr"]
    st = inference.state()

    warnings: list[str] = []
    yolo_url, cnn_raw, w1 = xai.yolo_cam(st.yolo, image) if st.yolo is not None else (None, None, "yolo not loaded")
    if w1:
        warnings.append(w1)
    detr_url, detr_raw, w2 = xai.detr_attention(st.rfdetr, image) if st.rfdetr is not None else (None, None, "rfdetr not loaded")
    if w2:
        warnings.append(w2)

    agreement_url: str | None = None
    if cnn_raw is not None and detr_raw is not None:
        agreement_url, w3 = xai.agreement_overlay(image, cnn_raw, detr_raw)
        if w3:
            warnings.append(w3)
    else:
        warnings.append("agreement map skipped (both CAM + attention required)")

    payload = {
        "scan_id": scan_id,
        "yolo_cam_url": yolo_url,
        "rfdetr_attention_url": detr_url,
        "agreement_map_url": agreement_url,
        "warnings": warnings,
    }
    update_session(scan_id, {"explain": payload})
    return payload
