"""Conformal Risk Control for object detection.

Two primitives:

  - Split-conformal per-coordinate bounding-box expansion so that
    P(true box contained in expanded box) >= 1 - alpha (Bonferroni over 4
    coordinates). See Andéol 2023, Timans 2024.

  - Conformal Risk Control (CRC, Angelopoulos 2022) with a monotone loss
    (False-Negative Rate) to choose the largest score threshold `lambda`
    such that E[FNR(lambda)] <= alpha on future data with high probability.

These are patent-strong because they provide distribution-free coverage
guarantees on top of any base detector without retraining.
"""

from .split_conformal import SplitConformalBox, fit_split_conformal
from .risk_control import ConformalRiskController, fit_recall_bound
from .adaptive import AdaptiveConformalBox

__all__ = [
    "SplitConformalBox",
    "fit_split_conformal",
    "ConformalRiskController",
    "fit_recall_bound",
    "AdaptiveConformalBox",
]
