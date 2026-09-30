"""Isotonic regression on fused ensemble confidence.

After WBF + shadow-prior produces a `fused_conf` per candidate, this maps
`fused_conf -> P(TP | fused_conf, IoU >= 0.5)` using isotonic regression
fit on a held-out validation split (ideally a different fold from that used
for temperature scaling to avoid double-dipping).

Wraps `sklearn.isotonic.IsotonicRegression` with save/load convenience.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError as e:  # pragma: no cover
    IsotonicRegression = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


@dataclass
class IsotonicFusion:
    """Isotonic mapping fused_conf -> calibrated probability."""

    model: object | None = None
    n_samples: int = 0
    metadata: dict = field(default_factory=dict)

    def fit(
        self,
        fused_conf: Sequence[float] | np.ndarray,
        is_tp: Sequence[int] | np.ndarray,
    ) -> "IsotonicFusion":
        if IsotonicRegression is None:
            raise RuntimeError(f"scikit-learn required for isotonic fusion: {_IMPORT_ERR}")
        x = np.asarray(fused_conf, dtype=np.float64).reshape(-1)
        y = np.asarray(is_tp, dtype=np.float64).reshape(-1)
        if x.shape != y.shape:
            raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(x, y)
        self.model = ir
        self.n_samples = int(x.size)
        return self

    def apply(self, fused_conf: np.ndarray | float) -> np.ndarray | float:
        if self.model is None:
            raise RuntimeError("IsotonicFusion not fitted")
        arr = np.asarray(fused_conf, dtype=np.float64)
        out = self.model.predict(arr.reshape(-1)).reshape(arr.shape)
        if np.isscalar(fused_conf):
            return float(out)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicFusion":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise RuntimeError(f"unexpected object in {path}: {type(obj)}")
        return obj
