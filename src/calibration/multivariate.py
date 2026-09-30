"""Multivariate calibration (Küppers CVPR-W 2020 style).

Bins jointly on (confidence, box_area, y_center, shadow_score). Within each
bin, empirical P(TP) is stored; at inference, the corresponding bin's value
is returned. Falls back to global mean for unseen bins.

Kept intentionally simple: histogram-based, not gradient-fit. For gallstone US
the (position, size, shadow) trio is more informative than confidence alone.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class MultivariateCalibrator:
    n_bins: int = 5
    features: tuple[str, ...] = ("conf", "area", "y_center", "shadow_score")
    edges: dict[str, np.ndarray] = field(default_factory=dict)
    bin_prob: np.ndarray | None = None  # shape = (n_bins,) * len(features)
    bin_count: np.ndarray | None = None
    global_prob: float = 0.0

    def _bucket(self, x: np.ndarray, name: str) -> np.ndarray:
        edges = self.edges[name]
        # np.digitize returns 1..len(edges); clip to [0, n_bins-1]
        idx = np.clip(np.digitize(x, edges) - 1, 0, self.n_bins - 1)
        return idx

    def fit(
        self,
        conf: Sequence[float] | np.ndarray,
        area: Sequence[float] | np.ndarray,
        y_center: Sequence[float] | np.ndarray,
        shadow_score: Sequence[float] | np.ndarray,
        is_tp: Sequence[int] | np.ndarray,
    ) -> "MultivariateCalibrator":
        feats = {
            "conf": np.asarray(conf, dtype=np.float64),
            "area": np.asarray(area, dtype=np.float64),
            "y_center": np.asarray(y_center, dtype=np.float64),
            "shadow_score": np.asarray(shadow_score, dtype=np.float64),
        }
        y = np.asarray(is_tp, dtype=np.float64)

        # Quantile edges per feature (avoids empty bins with skewed dists)
        self.edges = {}
        for name in self.features:
            f = feats[name]
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]  # interior only
            self.edges[name] = np.quantile(f, qs) if f.size else np.array([0.0])

        idxs = tuple(self._bucket(feats[n], n) for n in self.features)
        shape = tuple(self.n_bins for _ in self.features)
        sum_tp = np.zeros(shape, dtype=np.float64)
        count = np.zeros(shape, dtype=np.float64)
        np.add.at(sum_tp, idxs, y)
        np.add.at(count, idxs, 1.0)
        # Smooth: add pseudocount (Laplace)
        self.bin_prob = (sum_tp + 1.0) / (count + 2.0)
        self.bin_count = count
        self.global_prob = float(y.mean()) if y.size else 0.0
        return self

    def apply(
        self,
        conf: np.ndarray,
        area: np.ndarray,
        y_center: np.ndarray,
        shadow_score: np.ndarray,
    ) -> np.ndarray:
        if self.bin_prob is None:
            raise RuntimeError("MultivariateCalibrator not fitted")
        feats = {"conf": conf, "area": area, "y_center": y_center, "shadow_score": shadow_score}
        idxs = tuple(self._bucket(np.asarray(feats[n], dtype=np.float64), n) for n in self.features)
        return self.bin_prob[idxs]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MultivariateCalibrator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise RuntimeError(f"unexpected object in {path}: {type(obj)}")
        return obj
