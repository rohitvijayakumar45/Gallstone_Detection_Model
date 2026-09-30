"""Calibration scoring metrics.

  - ECE (Guo 2017)   : equal-width bins, gap-weighted mean.
  - adaptive-ECE     : equal-mass bins (quantile edges) - stable for skewed
                       distributions of predicted probabilities.
  - MCE              : max bin-gap.
  - Brier score      : mean squared error (probability vs 0/1 label).
  - NLL              : binary cross-entropy (per-sample average).
  - reliability_bins : returns per-bin (mean_conf, empirical_acc, count) for
                       plotting reliability diagrams.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1e-7


def _prep(probs: Sequence[float] | np.ndarray, labels: Sequence[int] | np.ndarray):
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.shape != y.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {y.shape}")
    return p, y


def expected_calibration_error(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    n_bins: int = 15,
) -> float:
    p, y = _prep(probs, labels)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = p.size
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        m = mask.sum()
        if m == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        total += (m / n) * abs(acc - conf)
    return float(total)


def adaptive_calibration_error(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    n_bins: int = 15,
) -> float:
    p, y = _prep(probs, labels)
    if p.size == 0:
        return 0.0
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    bins = np.array_split(np.arange(p.size), n_bins)
    total = 0.0
    n = p.size
    for idx in bins:
        if idx.size == 0:
            continue
        conf = p_sorted[idx].mean()
        acc = y_sorted[idx].mean()
        total += (idx.size / n) * abs(acc - conf)
    return float(total)


def maximum_calibration_error(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    n_bins: int = 15,
) -> float:
    p, y = _prep(probs, labels)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        worst = max(worst, abs(y[mask].mean() - p[mask].mean()))
    return float(worst)


def brier_score(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    p, y = _prep(probs, labels)
    if p.size == 0:
        return 0.0
    return float(((p - y) ** 2).mean())


def negative_log_likelihood(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    p, y = _prep(probs, labels)
    if p.size == 0:
        return 0.0
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def reliability_bins(
    probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Return per-bin dict for plotting a reliability diagram."""
    p, y = _prep(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        m = int(mask.sum())
        bins.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": m,
                "mean_conf": float(p[mask].mean()) if m else 0.0,
                "empirical_acc": float(y[mask].mean()) if m else 0.0,
            }
        )
    return bins
