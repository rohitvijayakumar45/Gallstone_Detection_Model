"""Conformal Risk Control (Angelopoulos 2022) for recall-bounded detection.

Chooses the largest score threshold `lambda*` such that empirical FNR on the
calibration set satisfies:

    (n / (n + 1)) * mean_FNR(lambda) + B / (n + 1) <= alpha

where B = 1 is the loss upper bound (FNR is bounded in [0, 1]). This gives a
distribution-free guarantee that expected future FNR <= alpha, hence
recall >= 1 - alpha in expectation.

FNR is computed per calibration image as fraction of GT boxes that are NOT
matched (IoU >= iou_thresh) by any prediction whose score >= lambda.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _iou_matrix(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> np.ndarray:
    if pred_xyxy.size == 0 or gt_xyxy.size == 0:
        return np.zeros((pred_xyxy.shape[0], gt_xyxy.shape[0]))
    p = pred_xyxy.astype(np.float64)
    g = gt_xyxy.astype(np.float64)
    x1 = np.maximum(p[:, None, 0], g[None, :, 0])
    y1 = np.maximum(p[:, None, 1], g[None, :, 1])
    x2 = np.minimum(p[:, None, 2], g[None, :, 2])
    y2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = ap[:, None] + ag[None, :] - inter + 1e-9
    return inter / union


def _fnr(pred_xyxy: np.ndarray, pred_scores: np.ndarray,
         gt_xyxy: np.ndarray, lam: float, iou_thresh: float) -> float:
    """Fraction of GT not matched by any pred with score >= lam."""
    if gt_xyxy.size == 0:
        return 0.0  # no positives; convention: FNR = 0
    keep = pred_scores >= lam
    if not keep.any():
        return 1.0
    ious = _iou_matrix(pred_xyxy[keep], gt_xyxy)
    matched_gt = (ious.max(axis=0) >= iou_thresh) if ious.size else np.zeros(len(gt_xyxy), dtype=bool)
    return float(1.0 - matched_gt.mean())


@dataclass
class ConformalRiskController:
    alpha: float = 0.05
    lambda_star: float = 0.0
    n_images: int = 0
    iou_thresh: float = 0.5

    def apply(self, pred_scores: np.ndarray | Sequence[float]) -> np.ndarray:
        s = np.asarray(pred_scores)
        return s >= self.lambda_star

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "lambda_star": self.lambda_star,
            "n_images": self.n_images,
            "iou_thresh": self.iou_thresh,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ConformalRiskController":
        return cls(**json.loads(Path(path).read_text()))


def fit_recall_bound(
    per_image_preds: list[dict],
    alpha: float = 0.05,
    iou_thresh: float = 0.5,
    lambda_grid: np.ndarray | None = None,
) -> ConformalRiskController:
    """Fit CRC threshold.

    per_image_preds: list of dicts with keys
        - boxes:  (m, 4) pixel bboxes
        - scores: (m,)
        - gt:     (k, 4) ground-truth pixel bboxes
    """
    if lambda_grid is None:
        # Descend from strict to lax; want LARGEST lambda satisfying bound.
        lambda_grid = np.linspace(0.99, 0.01, 99)

    n = len(per_image_preds)
    if n == 0:
        return ConformalRiskController(alpha=alpha, lambda_star=0.0, n_images=0,
                                       iou_thresh=iou_thresh)

    # We want largest lambda such that (n/(n+1))*mean_fnr + 1/(n+1) <= alpha.
    lambda_star = 0.0
    for lam in sorted(lambda_grid, reverse=True):
        fnrs = np.array(
            [_fnr(np.asarray(p["boxes"]), np.asarray(p["scores"]),
                  np.asarray(p["gt"]), float(lam), iou_thresh)
             for p in per_image_preds]
        )
        rhs = (n / (n + 1.0)) * float(fnrs.mean()) + 1.0 / (n + 1.0)
        if rhs <= alpha:
            lambda_star = float(lam)
            break

    return ConformalRiskController(
        alpha=float(alpha),
        lambda_star=lambda_star,
        n_images=int(n),
        iou_thresh=float(iou_thresh),
    )
