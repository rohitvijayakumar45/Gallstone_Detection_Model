"""Temperature scaling for per-detector confidence calibration.

Given per-candidate (logit, is_TP) pairs on a held-out validation split,
fit a scalar temperature T > 0 that minimises binary cross-entropy of
sigmoid(logit / T) against the label. Larger T softens overconfident
predictions; T < 1 sharpens under-confident ones.

For detectors that emit sigmoid confidences (typical YOLO / DETR), the logit
is `logit = log(p / (1 - p))`. `TemperatureScaler.fit_from_confidences`
handles this transform internally.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    if temperature <= 0:
        return 1e12
    p = _sigmoid(logits / temperature)
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Return T > 0 minimising binary cross-entropy on (logits, labels)."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError(f"shape mismatch: {logits.shape} vs {labels.shape}")
    if logits.size == 0:
        return 1.0

    res = minimize_scalar(
        _nll,
        args=(logits, labels),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


@dataclass
class TemperatureScaler:
    """Serializable per-detector temperature scaler."""

    temperature: float = 1.0
    n_samples: int = 0
    detector_name: str = "unknown"

    def apply_to_confidence(self, conf: np.ndarray | float) -> np.ndarray | float:
        conf_arr = np.asarray(conf, dtype=np.float64)
        p = _sigmoid(_logit(conf_arr) / self.temperature)
        if np.isscalar(conf):
            return float(p)
        return p

    def apply_to_logits(self, logits: np.ndarray | float) -> np.ndarray | float:
        logits_arr = np.asarray(logits, dtype=np.float64)
        p = _sigmoid(logits_arr / self.temperature)
        if np.isscalar(logits):
            return float(p)
        return p

    @classmethod
    def fit_from_confidences(
        cls,
        confidences: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        detector_name: str = "unknown",
    ) -> "TemperatureScaler":
        conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
        lbl = np.asarray(labels, dtype=np.float64).reshape(-1)
        logits = _logit(conf)
        t = fit_temperature(logits, lbl)
        return cls(temperature=t, n_samples=int(conf.size), detector_name=detector_name)

    @classmethod
    def fit_from_logits(
        cls,
        logits: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
        detector_name: str = "unknown",
    ) -> "TemperatureScaler":
        arr = np.asarray(logits, dtype=np.float64).reshape(-1)
        lbl = np.asarray(labels, dtype=np.float64).reshape(-1)
        t = fit_temperature(arr, lbl)
        return cls(temperature=t, n_samples=int(arr.size), detector_name=detector_name)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(
                json.dumps(
                    {
                        "temperature": self.temperature,
                        "n_samples": self.n_samples,
                        "detector_name": self.detector_name,
                    },
                    indent=2,
                )
            )
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        path = Path(path)
        if path.suffix == ".json":
            d = json.loads(path.read_text())
            return cls(**d)
        with open(path, "rb") as f:
            return pickle.load(f)
