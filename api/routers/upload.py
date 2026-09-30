"""Scan ingest: DICOM + JPG/PNG. Extracts real DICOM metadata."""

from __future__ import annotations

import base64
import io
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from api.services.session import get_session, put_session
from api.settings import settings
from src.preprocessing.dicom_meta import extract_dicom_meta

router = APIRouter(prefix="/upload", tags=["upload"])

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".dcm"}


def _decode_dicom(raw: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(raw))
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8) * 255.0
    img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    meta = extract_dicom_meta(raw)
    return img, meta


def _decode_standard(raw: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(pil)
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return img, {"is_dicom": False}


@router.post("")
async def upload(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename missing")
    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED:
        raise HTTPException(status_code=415, detail=f"unsupported format {ext}")

    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"file exceeds {settings.max_upload_mb} MB")

    try:
        if ext == ".dcm":
            image_bgr, meta = _decode_dicom(raw)
        else:
            image_bgr, meta = _decode_standard(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"decode failed: {exc}")

    h, w = image_bgr.shape[:2]
    scan_id = uuid.uuid4().hex[:12]

    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise HTTPException(status_code=500, detail="preview encode failed")
    preview_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    put_session(scan_id, {
        "image_bgr": image_bgr,
        "filename": file.filename,
        "width": w,
        "height": h,
        "preview_url": preview_url,
        "dicom_meta": meta,
    })

    return {
        "scan_id": scan_id,
        "filename": file.filename,
        "width": w,
        "height": h,
        "preview_url": preview_url,
        "dicom_meta": meta,
    }


@router.get("/{scan_id}")
async def upload_status(scan_id: str) -> dict:
    sess = get_session(scan_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="scan not found")
    return {
        "scan_id": scan_id,
        "filename": sess["filename"],
        "width": sess["width"],
        "height": sess["height"],
        "preview_url": sess["preview_url"],
        "dicom_meta": sess.get("dicom_meta", {}),
    }
