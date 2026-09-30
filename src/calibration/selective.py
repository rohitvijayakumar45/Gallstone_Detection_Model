"""Selective prediction / deferral.

Given calibrated per-candidate probabilities and TP/FP labels, learn a
threshold τ that maximises F1 on the "covered" subset at each target
coverage level c ∈ {0.7, 0.8, 0.9, 1.0}. Coverage = fraction of positives
not deferred.

At inference: if calibrated_conf >= τ → emit; else → verdict "review".
Also emits selective risk = 1 - precision on covered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


def _quantile_threshold(probs: np.ndarray, coverage: float) -> float:
    """Threshold such that fraction of samples with prob >= threshold ~= coverage."""
    if probs.size == 0:
        return 0.0
    coverage = float(np.clip(coverage, 0.0, 1.0))
    if coverage >= 1.0:
        return 0.0
    return float(np.quantile(probs, 1.0 - coverage))


def _f1_at(probs: np.ndarray, labels: np.ndarray, tau: float) -> tuple[float, float, float, float]:
    keep = probs >= tau
    if keep.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0
    tp = float(labels[keep].sum())
    fp = float(keep.sum() - tp)
    fn = float(labels.sum() - tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    coverage = float(keep.mean())
    return f1, prec, rec, coverage


@dataclass
class SelectiveThresholds:
    """Per-target-coverage thresholds + operating points."""

    entries: list[dict] = field(default_factory=list)

    def add(self, target_coverage: float, tau: float, metrics: dict) -> None:
        self.entries.append(
            {"target_coverage": float(target_coverage), "tau": float(tau), **metrics}
        )

    def tau_at_coverage(self, target_coverage: float) -> float:
        best = min(self.entries, key=lambda e: abs(e["target_coverage"] - target_coverage))
        return float(best["tau"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": self.entries}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "SelectiveThresholds":
        d = json.loads(Path(path).read_text())
        return cls(entries=list(d.get("entries", [])))


def fit_selective_thresholds(
    calibrated_probs: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    target_coverages: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0),
    tau_grid: np.ndarray | None = None,
) -> SelectiveThresholds:
    p = np.asarray(calibrated_probs, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return SelectiveThresholds()

    if tau_grid is None:
        tau_grid = np.linspace(0.01, 0.99, 99)

    out = SelectiveThresholds()
    for c in target_coverages:
        tau_start = _quantile_threshold(p, c)
        best = None
        # Search a narrow band around tau_start to trade F1 vs coverage
        band = tau_grid[(tau_grid >= max(0.0, tau_start - 0.15)) &
                        (tau_grid <= min(1.0, tau_start + 0.15))]
        for tau in np.concatenate([[tau_start], band]):
            f1, prec, rec, cov = _f1_at(p, y, float(tau))
            # Reject taus that undershoot coverage by > 5pt
            if cov < c - 0.05 and c < 1.0:
                continue
            score = f1
            if best is None or score > best[0]:
                best = (score, float(tau), {"f1": f1, "precision": prec,
                                             "recall": rec, "coverage": cov})
        if best is None:
            f1, prec, rec, cov = _f1_at(p, y, float(tau_start))
            best = (f1, float(tau_start), {"f1": f1, "precision": prec,
                                            "recall": rec, "coverage": cov})
        out.add(c, best[1], best[2])
    return out
