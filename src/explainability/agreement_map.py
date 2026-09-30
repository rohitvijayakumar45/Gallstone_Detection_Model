"""Cross-architecture attention-agreement map.

Combines a CNN feature-attribution map (Grad-CAM / EigenCAM on YOLO) with a
transformer decoder cross-attention map (RF-DETR) into a single per-pixel
"agreement" score. Novel component of the patent claim: high agreement
signals cross-architectural corroboration, low agreement flags regions only
one architecture "trusts".

We use a windowed Pearson correlation: for each output pixel, correlate a
local patch of the two normalised maps. This is less noisy than raw
element-wise multiplication and highlights spatial coherence.
"""

from __future__ import annotations

import cv2
import numpy as np


def _resize_norm(a: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if a.shape != target_hw:
        a = cv2.resize(a.astype(np.float32), (target_hw[1], target_hw[0]),
                       interpolation=cv2.INTER_CUBIC)
    a = a.astype(np.float32)
    a -= a.min()
    m = a.max()
    if m > 0:
        a /= m
    return a


def windowed_pearson(a: np.ndarray, b: np.ndarray, k: int = 15) -> np.ndarray:
    """Per-pixel Pearson r using a k×k box filter for local moments.

    Both inputs must be same-shape float in [0, 1]. Returns per-pixel r in
    [-1, 1]. Kernel size k should be odd.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if k % 2 == 0:
        k += 1
    kern = (k, k)
    ma = cv2.boxFilter(a, ddepth=cv2.CV_32F, ksize=kern)
    mb = cv2.boxFilter(b, ddepth=cv2.CV_32F, ksize=kern)
    maa = cv2.boxFilter(a * a, ddepth=cv2.CV_32F, ksize=kern)
    mbb = cv2.boxFilter(b * b, ddepth=cv2.CV_32F, ksize=kern)
    mab = cv2.boxFilter(a * b, ddepth=cv2.CV_32F, ksize=kern)
    cov = mab - ma * mb
    var_a = np.maximum(maa - ma * ma, 0.0)
    var_b = np.maximum(mbb - mb * mb, 0.0)
    denom = np.sqrt(var_a * var_b) + 1e-6
    r = cov / denom
    return np.clip(r, -1.0, 1.0).astype(np.float32)


def agreement_map(
    cnn_cam: np.ndarray,
    detr_attn: np.ndarray,
    target_hw: tuple[int, int] | None = None,
    window: int = 15,
) -> np.ndarray:
    """Return agreement map in [0, 1] at `target_hw` resolution."""
    if target_hw is None:
        target_hw = cnn_cam.shape[:2]
    a = _resize_norm(cnn_cam, target_hw)
    b = _resize_norm(detr_attn, target_hw)
    r = windowed_pearson(a, b, k=window)
    # Map from [-1, 1] to [0, 1] via (r + 1) / 2; only positive agreement is
    # clinically useful, so we clamp negatives to 0.5 baseline.
    agree = np.where(r > 0, r, 0.0)
    return agree.astype(np.float32)


def render_overlay(
    image_bgr: np.ndarray,
    agree: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Render agreement map as a colored overlay on the source image."""
    h, w = image_bgr.shape[:2]
    if agree.shape != (h, w):
        agree = cv2.resize(agree, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = (agree * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, colormap)
    return cv2.addWeighted(colored, alpha, image_bgr, 1.0 - alpha, 0.0)
