"""
main.py
-------
FastAPI backend for GallScan AI v3 — YOLO26 instance segmentation.

Endpoints
---------
GET  /api/health      liveness probe
POST /api/detect      full pipeline: inference + 3 explainability outputs
"""

import io
import os
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from model.detector import run_inference, get_model
from model.explainability import compute_all_explainability
from utils.image_utils import load_image, SUPPORTED_EXTENSIONS
from utils.report import build_report

# ── Initialisation ───────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GallScan AI v3",
    description="YOLO26 instance segmentation + EigenCAM + Prediction Stability",
    version="3.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = os.getenv("DEVICE", "cpu")
CONF_THRESH = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
MAX_MB     = 20


# ── Startup: pre-load model ──────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    weights = Path(os.getenv("WEIGHTS_PATH", "weights/gallstone_seg.pt"))
    if not weights.exists():
        logger.warning(
            "⚠  Model weights not found at '%s'. "
            "The /api/detect endpoint will return 503 until weights are available. "
            "Run  python train.py  to train the model.",
            weights,
        )
    else:
        try:
            get_model()
        except Exception as exc:
            logger.error("Failed to pre-load model: %s", exc)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["Health"])
async def health():
    weights_ok = Path(os.getenv("WEIGHTS_PATH", "weights/gallstone_seg.pt")).exists()
    return {
        "status":      "ok",
        "model":       "YOLO26-seg",
        "device":      DEVICE,
        "weights_ok":  weights_ok,
    }


@app.post("/api/detect", tags=["Detection"])
async def detect(file: UploadFile = File(...)):
    """
    Accept an ultrasound image (or DICOM), run YOLO26-seg inference,
    compute three explainability outputs, and return a structured result.
    """

    # ── 1. Validate ──────────────────────────────────────────────────────────
    ext = Path(file.filename or "upload.jpg").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    raw = await file.read()
    mb  = len(raw) / (1024 ** 2)
    if mb > MAX_MB:
        raise HTTPException(413, f"File too large ({mb:.1f} MB). Max {MAX_MB} MB.")

    # ── 2. Decode image ──────────────────────────────────────────────────────
    try:
        pil_image = load_image(raw, file.filename or "")
    except (ValueError, ImportError) as exc:
        raise HTTPException(400, str(exc))

    W, H = pil_image.size
    logger.info("Image loaded: %dx%d px  %.2f MB  [%s]", W, H, mb, file.filename)

    # ── 3. Check weights ─────────────────────────────────────────────────────
    weights = Path(os.getenv("WEIGHTS_PATH", "weights/gallstone_seg.pt"))
    if not weights.exists():
        raise HTTPException(
            503,
            "Model weights not found. "
            "Train the model first with  python train.py  "
            "or place fine-tuned weights at  backend/weights/gallstone_seg.pt",
        )

    # ── 4. Save to temp file and run inference ───────────────────────────────
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            pil_image.save(tmp, format="JPEG", quality=95)
            tmp_path = tmp.name

        det = run_inference(tmp_path, conf_threshold=CONF_THRESH, device=DEVICE)

    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(502, f"Inference error: {exc}")

    # ── 5. Compute explainability + stability-adjusted confidence ────────────
    model = get_model()
    stability_conf = 0.0

    try:
        eigencam_url, segmask_url, stability_url, stability_conf = \
            compute_all_explainability(
                model, tmp_path, det["result_obj"], pil_image, n_passes=10
            )
    except Exception as exc:
        logger.warning("Explainability failed: %s", exc, exc_info=True)
        eigencam_url = segmask_url = stability_url = ""

    # ── 6. Build report using best available confidence ───────────────────────
    # If the model detected something but primary-pass confidence is low,
    # use the cross-perturbation confidence when it is higher — it reflects
    # how consistently the model fires on that region across 10 noisy passes.
    primary_conf = max((b["confidence"] for b in det["boxes"]), default=0.0)
    adjusted_conf = max(primary_conf, stability_conf) if det["boxes"] else primary_conf

    if adjusted_conf != primary_conf and det["boxes"]:
        logger.info(
            "Confidence adjusted: primary=%.1f%% → stability=%.1f%%",
            primary_conf * 100, adjusted_conf * 100,
        )
        # Patch the top box confidence so report.py sees the adjusted value
        for b in det["boxes"]:
            b["_primary_confidence"] = b["confidence"]
        det["boxes"][0]["confidence"] = adjusted_conf

    report = build_report(det["boxes"], det["orig_shape"])

    # ── 7. Clean up temp file ────────────────────────────────────────────────
    if tmp_path:
        import os as _os
        try: _os.unlink(tmp_path)
        except OSError: pass

    # ── 8. Return ─────────────────────────────────────────────────────────────
    return JSONResponse({
        **report,
        "eigencam_url":        eigencam_url,
        "segmask_url":         segmask_url,
        "stability_url":       stability_url,
        "stability_confidence": stability_conf,
        "image_width":         W,
        "image_height":        H,
    })
