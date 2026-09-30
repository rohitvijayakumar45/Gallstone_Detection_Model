from __future__ import annotations

from fastapi import APIRouter

from api.services import inference
from api.settings import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    st = inference.state()
    return {
        "status": "healthy" if (st.yolo or st.rfdetr) else "degraded",
        "yolo_loaded": st.yolo is not None,
        "rfdetr_loaded": st.rfdetr is not None,
        "calibration_ready": all(x is not None for x in (
            st.calibration_yolo, st.calibration_rfdetr, st.isotonic_fusion
        )),
        "conformal_ready": st.conformal_qhat is not None,
        "device": st.device,
        "settings_summary": settings.summary(),
    }
