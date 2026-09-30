"""GallStone AI unified FastAPI backend.

Wires:
  - upload      (DICOM + JPG ingest, real DICOM metadata extraction)
  - predict     (WBF + shadow prior + temperature scaling + isotonic +
                 split-conformal + CRC + selective verdict)
  - explain     (Grad-CAM YOLO + DETR cross-attention + agreement map)
  - report      (real-unit measurements + honest impression)
  - metrics     (pre-computed benchmark / calibration / conformal reports)
  - health

Models + calibration + conformal artefacts load in the lifespan startup
context. Missing artefacts degrade gracefully.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.routers import explain, health, metrics, predict, report, upload
from api.services import inference
from api.settings import settings


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print("[startup] initialising models + calibration + conformal artefacts")
    st = inference.initialise(settings)
    print(f"[startup] yolo={'ok' if st.yolo else 'MISSING'}  "
          f"rfdetr={'ok' if st.rfdetr else 'MISSING'}  device={st.device}")
    print(f"[startup] calibration_ready="
          f"{all(x is not None for x in (st.calibration_yolo, st.calibration_rfdetr, st.isotonic_fusion))}  "
          f"conformal_ready={st.conformal_qhat is not None}")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title="GallStone AI - Patent-Grade Backend",
    description=(
        "Dual-detector ensemble (YOLOv26L + RF-DETR) with shadow-verified WBF, "
        "two-stage post-hoc calibration, split-conformal risk control, and "
        "cross-architecture attention agreement XAI."
    ),
    version="6.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All routers under /api prefix so the Vite proxy just forwards /api/*
API = "/api"
app.include_router(health.router, prefix=API)
app.include_router(upload.router, prefix=API)
app.include_router(predict.router, prefix=API)
app.include_router(explain.router, prefix=API)
app.include_router(report.router, prefix=API)
app.include_router(metrics.router, prefix=API)


@app.get("/")
async def root() -> dict:
    return {
        "service": "gallstone-ai",
        "version": app.version,
        "docs": "/docs",
        "api_prefix": API,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
