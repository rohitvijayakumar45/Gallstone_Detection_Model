"""MC-dropout uncertainty for detector outputs.

We approximate aleatoric+epistemic uncertainty per candidate by running N
stochastic forward passes with dropout active. For DETR-family models the
classification head has dropout that is disabled at eval; enabling it and
running N passes yields a distribution of confidences whose spread is a
usable uncertainty proxy at zero training cost.

For fully deterministic detectors (no dropout in the head), the estimator
falls back to input-perturbation Monte-Carlo: N passes with small Gaussian
noise + brightness jitter. Standard-deviation of matched-box confidences
across passes = aleatoric proxy; box coordinate jitter = epistemic proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np


def _iou_greedy_match(anchor: np.ndarray, others: np.ndarray) -> int:
    if others.size == 0:
        return -1
    x1 = np.maximum(anchor[0], others[:, 0])
    y1 = np.maximum(anchor[1], others[:, 1])
    x2 = np.minimum(anchor[2], others[:, 2])
    y2 = np.minimum(anchor[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    iou = inter / (a + b - inter + 1e-9)
    best = int(iou.argmax())
    return best if iou[best] >= 0.3 else -1


@dataclass
class UncertaintyResult:
    aleatoric: np.ndarray   # shape (n_anchors,)  std of confidences
    epistemic: np.ndarray   # shape (n_anchors,)  std of box-center jitter
    flag_for_review: np.ndarray  # shape (n_anchors,)  bool


class UncertaintyEstimator:
    """Perturbation-based Monte-Carlo uncertainty for any detector.

    Compatible with YOLO + RF-DETR wrappers via a `predict_fn(image_bgr)`
    that returns (boxes_pix (n,4), scores (n,)). N passes with small
    additive Gaussian noise + brightness jitter approximate MC dropout when
    the model has no dropout in inference.
    """

    def __init__(self, n_samples: int = 15, noise_sigma: float = 4.0,
                 brightness_jitter: float = 0.05, review_threshold: float = 0.15):
        self.n_samples = int(n_samples)
        self.noise_sigma = float(noise_sigma)
        self.brightness_jitter = float(brightness_jitter)
        self.review_threshold = float(review_threshold)
        self.rng = np.random.default_rng(0)

    def estimate(
        self,
        image_bgr: np.ndarray,
        anchor_boxes: np.ndarray,   # (n_anchors, 4) in pixel space
        predict_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> UncertaintyResult:
        n = len(anchor_boxes)
        if n == 0:
            empty = np.empty(0, dtype=np.float32)
            return UncertaintyResult(empty, empty, np.zeros(0, dtype=bool))

        confs = np.full((self.n_samples, n), np.nan, dtype=np.float32)
        centres = np.full((self.n_samples, n, 2), np.nan, dtype=np.float32)

        for k in range(self.n_samples):
            noise = self.rng.normal(0, self.noise_sigma, image_bgr.shape).astype(np.float32)
            bright = 1.0 + float(self.rng.uniform(-self.brightness_jitter, self.brightness_jitter))
            perturbed = np.clip(image_bgr.astype(np.float32) * bright + noise, 0, 255).astype(np.uint8)
            boxes_k, scores_k = predict_fn(perturbed)
            if len(boxes_k) == 0:
                continue
            boxes_k = np.asarray(boxes_k, dtype=np.float32)
            scores_k = np.asarray(scores_k, dtype=np.float32)
            for j, anchor in enumerate(anchor_boxes):
                m = _iou_greedy_match(np.asarray(anchor, dtype=np.float32), boxes_k)
                if m < 0:
                    continue
                confs[k, j] = scores_k[m]
                cx = 0.5 * (boxes_k[m, 0] + boxes_k[m, 2])
                cy = 0.5 * (boxes_k[m, 1] + boxes_k[m, 3])
                centres[k, j] = (cx, cy)

        # Aleatoric = std of matched confidences (0 where no matches)
        aleatoric = np.zeros(n, dtype=np.float32)
        epistemic = np.zeros(n, dtype=np.float32)
        for j in range(n):
            c = confs[:, j]
            c = c[~np.isnan(c)]
            if c.size >= 2:
                aleatoric[j] = float(c.std(ddof=0))
            cx = centres[:, j, 0]
            cy = centres[:, j, 1]
            cx = cx[~np.isnan(cx)]
            cy = cy[~np.isnan(cy)]
            if cx.size >= 2:
                epistemic[j] = float(np.hypot(cx.std(ddof=0), cy.std(ddof=0)))

        flag = aleatoric > self.review_threshold
        return UncertaintyResult(aleatoric, epistemic, flag)
