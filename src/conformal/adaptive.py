"""Adaptive / streaming conformal box expansion.

Maintains a running list of nonconformity residuals so that quantiles refresh
as new labelled examples arrive. Useful in a clinical deployment where
occasional radiologist verdicts stream back and the coverage guarantee should
adjust to distribution drift.

Not a full ACI (Gibbs & Candès 2021); a windowed empirical quantile with a
maximum buffer size and Bonferroni correction across coordinates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from .split_conformal import SplitConformalBox, _finite_quantile


@dataclass
class AdaptiveConformalBox:
    alpha: float = 0.05
    max_buffer: int = 1000
    bonferroni: bool = True
    _rx1: Deque[float] = field(default_factory=deque)
    _ry1: Deque[float] = field(default_factory=deque)
    _rx2: Deque[float] = field(default_factory=deque)
    _ry2: Deque[float] = field(default_factory=deque)

    def observe(self, pred_box_norm: np.ndarray, gt_box_norm: np.ndarray) -> None:
        p = np.asarray(pred_box_norm, dtype=np.float64).reshape(4)
        g = np.asarray(gt_box_norm, dtype=np.float64).reshape(4)
        vals = (
            max(0.0, p[0] - g[0]),
            max(0.0, p[1] - g[1]),
            max(0.0, g[2] - p[2]),
            max(0.0, g[3] - p[3]),
        )
        for buf, v in zip((self._rx1, self._ry1, self._rx2, self._ry2), vals):
            buf.append(float(v))
            while len(buf) > self.max_buffer:
                buf.popleft()

    def snapshot(self) -> SplitConformalBox:
        eff_alpha = self.alpha / 4.0 if self.bonferroni else self.alpha
        q = 1.0 - eff_alpha
        return SplitConformalBox(
            alpha=self.alpha,
            q_x1=_finite_quantile(np.asarray(self._rx1), q),
            q_y1=_finite_quantile(np.asarray(self._ry1), q),
            q_x2=_finite_quantile(np.asarray(self._rx2), q),
            q_y2=_finite_quantile(np.asarray(self._ry2), q),
            n_samples=len(self._rx1),
            bonferroni=self.bonferroni,
            metadata={"streaming": True, "buffer_cap": self.max_buffer},
        )

    def apply_norm(self, boxes_norm: np.ndarray) -> np.ndarray:
        return self.snapshot().apply_norm(boxes_norm)
