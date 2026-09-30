"""Polygon-aware YOLO label loader.

The dataset labels store polygons (class x1 y1 x2 y2 ... xn yn, all normalized
in [0,1]), not axis-aligned bbox rows (class cx cy w h). Prior evaluation
scripts assumed bbox rows and read the first four polygon coordinates as
cx/cy/w/h, producing garbage boxes and mAP == 0 across the board.

This module provides:
  - load_polygon_labels(label_path): yields lists of (class_id, polygon_xy) with
    polygon_xy shape (n, 2) normalized in [0,1].
  - polygon_to_bbox(polygon_xy): axis-aligned bbox in normalized [x1,y1,x2,y2].
  - load_bboxes(label_path, img_wh): pixel bboxes for detection eval.
  - iter_split(split_dir): iterate (image_bgr, pixel_bboxes, classes) over a
    YOLO-style split folder ({split}/images, {split}/labels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np


def load_polygon_labels(label_path: Path) -> List[Tuple[int, np.ndarray]]:
    """Return list of (class_id, polygon_xy [n,2] normalized)."""
    out: List[Tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return out
    with open(label_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
            # Ignore rows that are neither bbox (4 coords) nor a valid polygon
            # (>=6 coords, even count).
            if len(coords) == 4:
                # Legacy YOLO bbox row: cx cy w h -> convert to a 4-vertex polygon
                cx, cy, w, h = coords
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
                poly = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                poly = np.array(coords, dtype=np.float32).reshape(-1, 2)
            else:
                continue
            out.append((cls, poly))
    return out


def polygon_to_bbox(polygon_xy: np.ndarray) -> np.ndarray:
    """Axis-aligned bbox from a normalized polygon, clipped to [0,1]."""
    x1, y1 = polygon_xy[:, 0].min(), polygon_xy[:, 1].min()
    x2, y2 = polygon_xy[:, 0].max(), polygon_xy[:, 1].max()
    return np.array(
        [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)], dtype=np.float32
    )


def load_bboxes(label_path: Path, img_wh: Tuple[int, int]) -> np.ndarray:
    """Load per-object pixel bboxes [x1,y1,x2,y2] shape (n,4)."""
    w, h = img_wh
    boxes = []
    for _cls, poly in load_polygon_labels(label_path):
        b = polygon_to_bbox(poly)
        boxes.append([b[0] * w, b[1] * h, b[2] * w, b[3] * h])
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def iter_split(split_dir: Path) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
    """Yield (stem, image_bgr, pixel_bboxes) for each image in a YOLO split."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"images/ dir missing under {split_dir}")
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        bboxes = load_bboxes(lbl_dir / f"{img_path.stem}.txt", (w, h))
        yield img_path.stem, img, bboxes
