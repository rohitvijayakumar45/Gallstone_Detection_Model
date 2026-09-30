"""Split-conformal box expansion.

Given a calibration set of matched (pred_box, gt_box) pairs, compute
per-coordinate nonconformity residuals and their quantile so that a new box
expanded by those quantiles contains the true box with marginal probability
>= 1 - alpha.

We treat each coordinate independently and apply a Bonferroni correction
(target coverage per coordinate = 1 - alpha/4) for a joint 4-D guarantee.

Residual per coordinate:
    r_{x1} = max(0, pred_x1 - gt_x1)   (we may have overshot left edge)
    r_{y1} = max(0, pred_y1 - gt_y1)
    r_{x2} = max(0, gt_x2 - pred_x2)   (we may have undershot right edge)
    r_{y2} = max(0, gt_y2 - pred_y2)

Expansion at inference:
    conformal_box = [pred_x1 - q_x1,
                     pred_y1 - q_y1,
                     pred_x2 + q_x2,
                     pred_y2 + q_y2]

Coordinates are stored in NORMALISED image space [0, 1] to keep the quantile
comparable across images of different resolutions. Call `apply_pixel` to
receive pixel-space boxes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def _finite_quantile(vals: np.ndarray, q: float) -> float:
    if vals.size == 0:
        return 0.0
    q = float(np.clip(q, 0.0, 1.0))
    # Small-sample conformal quantile: use ceil((n+1)*q)/n index
    n = vals.size
    k = int(np.ceil((n + 1) * q))
    k = min(max(k, 1), n)
    return float(np.sort(vals)[k - 1])


@dataclass
class SplitConformalBox:
    """Per-coordinate quantile expansion for detection boxes."""

    alpha: float = 0.05
    q_x1: float = 0.0
    q_y1: float = 0.0
    q_x2: float = 0.0
    q_y2: float = 0.0
    n_samples: int = 0
    bonferroni: bool = True
    metadata: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        pred_boxes_norm: np.ndarray,
        gt_boxes_norm: np.ndarray,
        alpha: float = 0.05,
        bonferroni: bool = True,
    ) -> "SplitConformalBox":
        """Fit quantiles.

        Args:
          pred_boxes_norm: (n, 4) predicted boxes in [x1,y1,x2,y2], normalised [0,1].
          gt_boxes_norm:   (n, 4) matched ground-truth boxes, same shape.
          alpha:           marginal miscoverage (target coverage = 1 - alpha).
          bonferroni:      if True, use alpha/4 per coordinate for joint 4-D
                           coverage; otherwise per-coordinate marginal.
        """
        p = np.asarray(pred_boxes_norm, dtype=np.float64)
        g = np.asarray(gt_boxes_norm, dtype=np.float64)
        if p.shape != g.shape or p.ndim != 2 or p.shape[1] != 4:
            raise ValueError(f"expected (n, 4) matched arrays; got {p.shape} / {g.shape}")

        r_x1 = np.maximum(0.0, p[:, 0] - g[:, 0])
        r_y1 = np.maximum(0.0, p[:, 1] - g[:, 1])
        r_x2 = np.maximum(0.0, g[:, 2] - p[:, 2])
        r_y2 = np.maximum(0.0, g[:, 3] - p[:, 3])

        eff_alpha = alpha / 4.0 if bonferroni else alpha
        target_q = 1.0 - eff_alpha
        return cls(
            alpha=float(alpha),
            q_x1=_finite_quantile(r_x1, target_q),
            q_y1=_finite_quantile(r_y1, target_q),
            q_x2=_finite_quantile(r_x2, target_q),
            q_y2=_finite_quantile(r_y2, target_q),
            n_samples=int(p.shape[0]),
            bonferroni=bool(bonferroni),
        )

    def apply_norm(self, boxes_norm: np.ndarray) -> np.ndarray:
        """Expand normalised boxes; clip to [0, 1]."""
        b = np.asarray(boxes_norm, dtype=np.float64)
        if b.size == 0:
            return b
        out = b.copy()
        out[..., 0] = np.clip(out[..., 0] - self.q_x1, 0.0, 1.0)
        out[..., 1] = np.clip(out[..., 1] - self.q_y1, 0.0, 1.0)
        out[..., 2] = np.clip(out[..., 2] + self.q_x2, 0.0, 1.0)
        out[..., 3] = np.clip(out[..., 3] + self.q_y2, 0.0, 1.0)
        return out

    def apply_pixel(self, boxes_pixel: np.ndarray, img_wh: tuple[int, int]) -> np.ndarray:
        w, h = img_wh
        b = np.asarray(boxes_pixel, dtype=np.float64)
        if b.size == 0:
            return b
        out = b.copy()
        out[..., 0] = np.clip(out[..., 0] - self.q_x1 * w, 0.0, float(w))
        out[..., 1] = np.clip(out[..., 1] - self.q_y1 * h, 0.0, float(h))
        out[..., 2] = np.clip(out[..., 2] + self.q_x2 * w, 0.0, float(w))
        out[..., 3] = np.clip(out[..., 3] + self.q_y2 * h, 0.0, float(h))
        return out

    def coverage(
        self, pred_boxes_norm: np.ndarray, gt_boxes_norm: np.ndarray
    ) -> float:
        """Empirical coverage on a held-out test set."""
        expanded = self.apply_norm(pred_boxes_norm)
        g = np.asarray(gt_boxes_norm, dtype=np.float64)
        contains = (
            (expanded[:, 0] <= g[:, 0])
            & (expanded[:, 1] <= g[:, 1])
            & (expanded[:, 2] >= g[:, 2])
            & (expanded[:, 3] >= g[:, 3])
        )
        return float(contains.mean()) if contains.size else 0.0

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "q_x1": self.q_x1,
            "q_y1": self.q_y1,
            "q_x2": self.q_x2,
            "q_y2": self.q_y2,
            "n_samples": self.n_samples,
            "bonferroni": self.bonferroni,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SplitConformalBox":
        d = json.loads(Path(path).read_text())
        return cls(**d)


def fit_split_conformal(
    pred_boxes_norm: np.ndarray,
    gt_boxes_norm: np.ndarray,
    alpha: float = 0.05,
    bonferroni: bool = True,
) -> SplitConformalBox:
    return SplitConformalBox.fit(pred_boxes_norm, gt_boxes_norm, alpha, bonferroni)
