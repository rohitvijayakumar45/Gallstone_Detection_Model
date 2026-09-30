"""Unit tests for conformal primitives. No model weights needed."""

from __future__ import annotations

import numpy as np
import pytest

from src.conformal import (
    AdaptiveConformalBox,
    ConformalRiskController,
    SplitConformalBox,
    fit_recall_bound,
    fit_split_conformal,
)


def _synthetic_boxes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(0.1, 0.6, (n, 2))
    wh = rng.uniform(0.05, 0.2, (n, 2))
    gt_boxes = np.concatenate([gt, gt + wh], axis=1)
    # Predictions with symmetric noise
    noise = rng.normal(0, 0.02, (n, 4))
    pred_boxes = np.clip(gt_boxes + noise, 0.0, 1.0)
    return pred_boxes.astype(np.float64), gt_boxes.astype(np.float64)


def test_split_conformal_approximate_coverage():
    preds, gts = _synthetic_boxes(n=500, seed=0)
    # Fit on first half, test on second half (proper split-conformal usage)
    scb = fit_split_conformal(preds[:250], gts[:250], alpha=0.10, bonferroni=True)
    cov = scb.coverage(preds[250:], gts[250:])
    # Bonferroni is conservative; expect coverage >= 1 - alpha with slack
    assert cov >= 0.85, f"coverage too low: {cov}"


def test_split_conformal_bonferroni_is_conservative():
    preds, gts = _synthetic_boxes(n=800, seed=1)
    scb_b = fit_split_conformal(preds[:400], gts[:400], alpha=0.10, bonferroni=True)
    scb_m = fit_split_conformal(preds[:400], gts[:400], alpha=0.10, bonferroni=False)
    # Bonferroni uses smaller effective alpha -> larger expansion -> larger q
    assert scb_b.q_x1 + scb_b.q_x2 + scb_b.q_y1 + scb_b.q_y2 >= \
           scb_m.q_x1 + scb_m.q_x2 + scb_m.q_y1 + scb_m.q_y2


def test_split_conformal_expand_clips_to_unit():
    scb = SplitConformalBox(q_x1=0.5, q_y1=0.5, q_x2=0.5, q_y2=0.5)
    boxes = np.array([[0.1, 0.1, 0.9, 0.9]])
    out = scb.apply_norm(boxes)
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert out[0, 2] == 1.0 and out[0, 3] == 1.0


def test_split_conformal_save_load(tmp_path):
    scb = SplitConformalBox(alpha=0.05, q_x1=0.01, q_y1=0.02, q_x2=0.03, q_y2=0.04,
                            n_samples=100, bonferroni=True)
    f = tmp_path / "conf.json"
    scb.save(f)
    loaded = SplitConformalBox.load(f)
    assert abs(loaded.q_x1 - 0.01) < 1e-9
    assert loaded.bonferroni is True


def test_crc_returns_valid_lambda():
    rng = np.random.default_rng(2)
    per_image = []
    for _ in range(50):
        n_gt = rng.integers(1, 4)
        gt = rng.uniform(0.1, 0.6, (n_gt, 2))
        gt = np.concatenate([gt, gt + rng.uniform(0.05, 0.15, (n_gt, 2))], axis=1)
        # Predictions include GT jittered + noise + spurious
        preds = np.concatenate([gt + rng.normal(0, 0.02, gt.shape),
                                 rng.uniform(0, 1, (5, 4))])
        scores = np.concatenate([rng.uniform(0.6, 0.99, n_gt),
                                 rng.uniform(0.1, 0.6, 5)])
        per_image.append({"boxes": preds, "scores": scores, "gt": gt})

    crc = fit_recall_bound(per_image, alpha=0.15)
    assert 0.0 <= crc.lambda_star <= 1.0
    assert crc.alpha == 0.15


def test_adaptive_conformal_grows_buffer_and_snapshots():
    ac = AdaptiveConformalBox(alpha=0.1, max_buffer=50)
    rng = np.random.default_rng(3)
    for _ in range(100):
        gt = np.array([0.2, 0.2, 0.5, 0.5])
        pred = gt + rng.normal(0, 0.02, 4)
        ac.observe(pred, gt)
    snap = ac.snapshot()
    # Buffer capped at 50
    assert snap.n_samples == 50
    # Non-trivial quantile after observations
    assert snap.q_x1 >= 0.0
