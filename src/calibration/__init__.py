"""Post-hoc confidence calibration.

Two-stage cascade:
  A. Per-detector temperature scaling on val split (minimises NLL).
  B. Post-fusion isotonic regression on a second val fold (maps raw fused
     confidence to P(TP | fused)).
  C. Optional multivariate calibration binning on (conf, box_area, y_center,
     shadow_score) - Küppers CVPR-W 2020 style.

Scoring: ECE / adaptive-ECE / MCE / Brier / NLL and reliability diagram data.

Reference:
  Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
  Küppers et al., "Multivariate Confidence Calibration for Object
    Detection", CVPRW 2020.
  Munir et al., "Cal-DETR: Calibrated Detection Transformer", 2023.
"""

from .temperature import TemperatureScaler, fit_temperature
from .isotonic_fusion import IsotonicFusion
from .multivariate import MultivariateCalibrator
from .selective import SelectiveThresholds, fit_selective_thresholds
from .ece_mce import (
    expected_calibration_error,
    adaptive_calibration_error,
    maximum_calibration_error,
    brier_score,
    negative_log_likelihood,
    reliability_bins,
)

__all__ = [
    "TemperatureScaler",
    "fit_temperature",
    "IsotonicFusion",
    "MultivariateCalibrator",
    "SelectiveThresholds",
    "fit_selective_thresholds",
    "expected_calibration_error",
    "adaptive_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "negative_log_likelihood",
    "reliability_bins",
]
