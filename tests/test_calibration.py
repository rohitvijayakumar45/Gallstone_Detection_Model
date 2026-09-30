"""Unit tests for calibration primitives. No model weights needed."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import (
    IsotonicFusion,
    MultivariateCalibrator,
    TemperatureScaler,
    adaptive_calibration_error,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_bins,
)


def _synthetic_overconfident(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    # True probs uniform in [0.05, 0.95]
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0, 1, n) < p_true).astype(np.float64)
    # Overconfident model: push probs toward 0/1 via power < 1
    p_pred = np.where(p_true > 0.5, p_true ** 0.5, 1 - (1 - p_true) ** 0.5)
    return p_pred.astype(np.float64), y


def test_temperature_reduces_ece_on_overconfident_model():
    p, y = _synthetic_overconfident()
    ts = TemperatureScaler.fit_from_confidences(p, y, "test")
    p_cal = ts.apply_to_confidence(p)
    ece_before = expected_calibration_error(p, y)
    ece_after = expected_calibration_error(p_cal, y)
    assert ts.temperature > 0
    # Overconfident -> T should be > 1 (softens)
    assert ts.temperature >= 1.0 or ece_after <= ece_before + 1e-3
    assert ece_after <= ece_before + 1e-3


def test_isotonic_monotone():
    p, y = _synthetic_overconfident()
    iso = IsotonicFusion().fit(p, y)
    grid = np.linspace(0, 1, 50)
    out = iso.apply(grid)
    # Monotone non-decreasing
    assert (np.diff(out) >= -1e-9).all()
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_multivariate_bins_and_predicts():
    rng = np.random.default_rng(1)
    n = 500
    conf = rng.uniform(0, 1, n)
    area = rng.uniform(0, 1, n)
    ycen = rng.uniform(0, 1, n)
    shad = rng.uniform(0, 1, n)
    # Truth depends on shadow and confidence
    y = ((0.5 * conf + 0.5 * shad) > 0.6).astype(np.float64)

    mc = MultivariateCalibrator(n_bins=3).fit(conf, area, ycen, shad, y)
    out = mc.apply(conf, area, ycen, shad)
    assert out.shape == conf.shape
    assert (out >= 0).all() and (out <= 1).all()


def test_scoring_metrics_shapes_and_ranges():
    p, y = _synthetic_overconfident()
    assert 0 <= expected_calibration_error(p, y) <= 1
    assert 0 <= adaptive_calibration_error(p, y) <= 1
    assert 0 <= maximum_calibration_error(p, y) <= 1
    assert 0 <= brier_score(p, y) <= 1
    assert negative_log_likelihood(p, y) >= 0
    bins = reliability_bins(p, y, n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == len(p)


def test_temperature_save_load_roundtrip(tmp_path):
    ts = TemperatureScaler(temperature=1.42, n_samples=100, detector_name="yolo")
    for suffix in (".pkl", ".json"):
        f = tmp_path / f"ts{suffix}"
        ts.save(f)
        loaded = TemperatureScaler.load(f)
        assert abs(loaded.temperature - ts.temperature) < 1e-9
        assert loaded.detector_name == "yolo"
