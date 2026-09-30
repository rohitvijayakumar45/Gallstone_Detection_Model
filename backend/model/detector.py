"""
detector.py
-----------
YOLO26 instance segmentation inference wrapper.

YOLO26 is Ultralytics' latest architecture featuring:
  - End-to-end NMS-free inference (faster, no post-processing step)
  - Simplified architecture (DFL removed for broader hardware compatibility)
  - MuSGD optimizer for training
  - Up to 43% faster on CPU vs predecessors
  - ProgLoss + STAL for improved small-object detection

API is identical to previous Ultralytics models — same result.boxes / result.masks.
"""

import os
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_model = None   # singleton


def get_model():
    """Load YOLO26-seg weights once and cache for subsequent calls."""
    global _model
    if _model is not None:
        return _model

    from ultralytics import YOLO

    weights = Path(os.getenv("WEIGHTS_PATH", "weights/gallstone_seg.pt"))
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{weights}'.\n"
            "• To train: run  python train.py\n"
            "• Or download pre-trained YOLO26 weights and fine-tune:\n"
            "      yolo segment train model=yolo26s-seg.pt data=dataset/data.yaml"
        )

    logger.info("Loading YOLO26-seg weights from %s", weights)
    _model = YOLO(str(weights))
    _model.fuse()   # fuse Conv+BN for faster inference
    logger.info("YOLO26 model loaded  (%s)", weights.name)
    return _model


def run_inference(
    image_path: str,
    conf_threshold: float = 0.25,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run YOLO26-seg inference on a saved image file.

    Parameters
    ----------
    image_path      : str   – path to the image on disk
    conf_threshold  : float – minimum confidence to include a detection
    device          : str   – "cpu", "cuda", or "mps"

    Returns
    -------
    dict:
        boxes       – list of detection dicts (see below)
        orig_shape  – (H, W) of the original image
        result_obj  – raw Ultralytics Result (passed to explainability module)
    """
    model = get_model()

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        device=device,
        save=False,
        verbose=False,
        retina_masks=True,   # full-resolution masks aligned to original image
    )

    result = results[0]
    H, W   = result.orig_shape
    boxes  = []

    for i, box in enumerate(result.boxes):
        entry = {
            "x":            float(box.xywh[0][0]),   # centre x  (pixels)
            "y":            float(box.xywh[0][1]),   # centre y  (pixels)
            "width":        float(box.xywh[0][2]),   # box width
            "height":       float(box.xywh[0][3]),   # box height
            "confidence":   float(box.conf[0]),
            "class_name":   result.names[int(box.cls[0])],
            "mask_polygon": [],
            "mask_area_px": 0,
        }

        # Instance segmentation mask
        if result.masks is not None and i < len(result.masks.xy):
            poly = result.masks.xy[i]           # (N, 2) float array
            if poly.size > 0:
                entry["mask_polygon"] = poly.tolist()
                binary = result.masks.data[i].cpu().numpy() > 0.5
                entry["mask_area_px"] = int(binary.sum())

        boxes.append(entry)

    return {
        "boxes":      boxes,
        "orig_shape": (H, W),
        "result_obj": result,
    }
