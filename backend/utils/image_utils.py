"""
image_utils.py
--------------
Image loading and pre-processing utilities.
Handles JPEG, PNG, WebP, TIFF, and DICOM (.dcm) files.
"""

import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "application/dicom",
    "application/octet-stream",   # some browsers send .dcm as this
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".dcm"}


def load_image(raw_bytes: bytes, filename: str = "") -> Image.Image:
    """
    Load image bytes into a PIL Image (RGB).
    Handles standard image formats and DICOM files transparently.

    Parameters
    ----------
    raw_bytes : bytes
        Raw file bytes as received from the upload.
    filename : str
        Original filename — used to detect DICOM by extension.

    Returns
    -------
    PIL.Image.Image  (mode="RGB")

    Raises
    ------
    ValueError  if the file cannot be decoded.
    """
    ext = Path(filename).suffix.lower()

    # ── DICOM ───────────────────────────────────────────────────────────────
    if ext == ".dcm" or _is_dicom(raw_bytes):
        return _load_dicom(raw_bytes)

    # ── Standard image formats ───────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        return img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image file: {exc}") from exc


def _is_dicom(data: bytes) -> bool:
    """Check for the DICOM magic bytes at offset 128."""
    return len(data) > 132 and data[128:132] == b"DICM"


def _load_dicom(raw_bytes: bytes) -> Image.Image:
    """
    Extract pixel data from a DICOM file and return as an RGB PIL Image.
    Applies window/level normalisation using the embedded tags when available.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom is required for DICOM support. Run: pip install pydicom"
        )

    ds = pydicom.dcmread(io.BytesIO(raw_bytes))

    pixels = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept (Hounsfield units for CT, raw for US)
    slope     = float(getattr(ds, "RescaleSlope",     1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    pixels    = pixels * slope + intercept

    # Apply window / level if present
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)
    if wc is not None and ww is not None:
        wc = float(wc[0] if hasattr(wc, "__len__") else wc)
        ww = float(ww[0] if hasattr(ww, "__len__") else ww)
        lo = wc - ww / 2
        hi = wc + ww / 2
        pixels = np.clip(pixels, lo, hi)

    # Normalise to [0, 255]
    lo, hi = pixels.min(), pixels.max()
    if hi > lo:
        pixels = (pixels - lo) / (hi - lo) * 255
    else:
        pixels = np.zeros_like(pixels)

    pixels = pixels.astype(np.uint8)

    # Convert greyscale → RGB
    if pixels.ndim == 2:
        img = Image.fromarray(pixels, mode="L").convert("RGB")
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        img = Image.fromarray(pixels, mode="RGB")
    else:
        img = Image.fromarray(pixels[:, :, 0], mode="L").convert("RGB")

    logger.info("DICOM decoded: %dx%d px", img.width, img.height)
    return img
