# Draft Patent Claims — GallStone AI v6

> This is an engineer-drafted starting point for a provisional filing. It
> MUST be reviewed by qualified patent counsel before submission. Numeric
> defaults are illustrative; ranges are what should be claimed.

---

## Independent Claim 1 (Method)

A computer-implemented method for detecting gallstones in an abdominal
ultrasound image comprising:

**(a) Preprocessing** the input image by:
  (i) contrast-limited adaptive histogram equalization applied to a
      grayscale representation of the image;
  (ii) speckle reduction by a filter selected from a non-local-means filter
       and a bilateral filter;
  (iii) preserving posterior acoustic shadow contrast by clamping pixels
        below a lower percentile of the pre-equalized image intensity
        distribution to their pre-equalized values;

**(b) Producing two sets of candidate bounding boxes** by inferring
    respectively:
  (i) with a single-stage convolutional detector; and
  (ii) with a transformer-based end-to-end detector having an object-query
       decoder;
  wherein at least one of the two detectors is trained under polygon-level
  segmentation supervision;

**(c) Computing, for each candidate bounding box, a posterior-acoustic-shadow
    score** as a monotone function of the ratio of the mean pixel intensity
    of a rectangular region immediately posterior to the candidate and the
    mean pixel intensity of one or more lateral reference regions at
    equivalent axial depth;

**(d) Fusing** the two candidate sets by weighted-box-fusion, wherein per-
    detector weights are derived from a monotone function of per-detector
    validation mean-average-precision and per-candidate weights are
    modulated by the posterior-acoustic-shadow score of (c);

**(e) Calibrating** the fused per-candidate confidence in two stages:
  (i) per-detector temperature scaling by a scalar `T > 0` fit to minimize
      negative log-likelihood of correctness on a held-out calibration
      split; and
  (ii) post-fusion isotonic regression fit on a second calibration fold that
       maps shadow-adjusted fused confidence to an empirical true-positive
       probability;

**(f) Producing, for each detected candidate, an expanded bounding box** by
    a split-conformal procedure comprising:
  (i) computing per-coordinate nonconformity residuals between predicted
      and ground-truth boxes on a calibration split;
  (ii) taking a Bonferroni-corrected `1 - α/4` quantile of said residuals
       per coordinate; and
  (iii) expanding the predicted box outward by said quantiles;
  such that the expanded box contains the true lesion box with marginal
  probability at least `1 - α`;

**(g) Filtering** the emitted detections at inference time by a score
    threshold `λ*` chosen from a calibration split by conformal risk control
    such that expected false-negative rate is bounded by `α`, thereby
    providing an expected recall guarantee of at least `1 - α`; and

**(h) Emitting** a structured report for each detection comprising: the
    point-estimate bounding box, the calibrated true-positive probability,
    the expanded conformal bounding box, the miscoverage parameter `α`, and
    a selective-prediction verdict from the set `{detected, review, clear}`
    determined by comparison of the calibrated probability to a learned
    per-coverage threshold.

---

## Dependent Claims

### Claim 2 — Real-unit measurements
The method of claim 1, further comprising extracting a `PixelSpacing` field
from a DICOM header associated with the input image and reporting lesion
area in square millimetres as the pixel-area of the fused bounding box
multiplied by the square of said `PixelSpacing`.

### Claim 3 — Aleatoric uncertainty
The method of claim 1, further comprising running `N ≥ 5` stochastic
forward passes through the transformer-based detector with dropout active
at the classification head, and reporting the standard deviation of
per-pass confidences as an aleatoric-uncertainty proxy per candidate.

### Claim 4 — Cross-architecture attention-agreement map
The method of claim 1, further comprising computing a per-pixel Pearson
correlation, over a local window, between a gradient- or feature-based
attribution map produced by the convolutional detector and a
cross-attention map produced by a decoder layer of the transformer
detector, thereby producing a per-pixel agreement score.

### Claim 5 — Selective prediction
The method of claim 1, wherein the selective-prediction verdict is
determined by comparing the calibrated probability against a threshold `τ`
learned on the calibration split to maximise a scoring function selected
from F1 score and precision-at-target-recall subject to a target
coverage constraint.

### Claim 6 — Multivariate calibration
The method of claim 1, wherein the calibration step further comprises a
histogram-based multivariate calibrator that partitions the calibration
data jointly on features `(confidence, box area, vertical position,
posterior-acoustic-shadow score)` and, at inference, returns the
empirical true-positive probability of the corresponding partition cell.

### Claim 7 — Cross-architecture consistency loss
The method of claim 1, wherein at least one detector is trained with an
auxiliary loss term proportional to the divergence — computed after
Hungarian matching on intersection-over-union — between per-anchor outputs
of the two detectors.

### Claim 8 — Streaming conformal recalibration
The method of claim 1, wherein the split-conformal quantiles are updated
online from a bounded first-in-first-out buffer of newly observed
(prediction, ground-truth) pairs, and the emitted expanded boxes reflect
the most recent quantile snapshot.

### Claim 9 — Failure-mode routing
The method of claim 1, further comprising a classifier that assigns each
candidate to one or more failure modes selected from
`{bowel-gas artefact, polyp, sludge, wall-thickening, non-shadowing stone,
contracted gallbladder}` and adjusts the selective-prediction threshold
accordingly.

### Claim 10 — Provenance annotation
The method of claim 1, wherein each numeric field of the emitted structured
report is accompanied by a provenance identifier from the set `{raw,
temperature-scaled, isotonic-fused, conformal, selective, multivariate}`.

### Claim 11 — Multi-scale test-time augmentation
The method of claim 1, wherein each detector runs at a plurality of input
scales prior to weighted-box-fusion, and per-scale weights are jointly
optimised on the calibration split.

### Claim 12 — System claim
A system comprising:
  (i) a backend server implementing the method of claim 1 and exposing
      HTTP endpoints for image upload, prediction, explanation, structured
      report, and pre-computed benchmark tables; and
  (ii) a browser-based clinical workstation displaying the point-estimate
       bounding box, calibrated probability, expanded conformal bounding
       box, attention-agreement map, and selective-prediction verdict as
       independently toggleable overlays on the input image.

### Claim 13 — Medium claim
A non-transitory computer-readable storage medium storing instructions
that, when executed by one or more processors, cause the processors to
perform the method of claim 1.

### Claim 14 — External-institution generalization
The method of claim 1, wherein the calibration and conformal artefacts are
re-fit on a data split originating from an institution different from that
which supplied the training data, prior to deployment at said different
institution.

---

## Notes for counsel

- The **core novelty** for gallstone US is the combination of
  posterior-acoustic-shadow prior, two-stage post-hoc calibration, and
  split-conformal box expansion + CRC recall bound in a single
  polygon-supervised dual-detector system. No single cited prior art
  teaches all elements together.
- **Independent claim style**: method (1) + system (12) + medium (13) is
  the standard triad; consider adding a training-time claim once the
  cross-architecture consistency loss (claim 7) is empirically validated.
- Consider **narrowing** ranges post-empirical: e.g., quantile α ∈ [0.01,
  0.10], WBF IoU threshold ∈ [0.3, 0.7], shadow-depth `D ∈ [50, 200]` px,
  window size for agreement Pearson `∈ [11, 25]`.
- **Prior-art distinguishers** worth naming: Solovyev 2021 (WBF alone),
  Munir 2023 Cal-DETR (single-model calibration), Angelopoulos 2022 (CRC
  generic, no ensemble/shadow), Han 2025 SM-WBF (HCC US, no calibration,
  no conformal), E-DETR 2025 (evidential single model). See
  [PRIOR_ART.md](PRIOR_ART.md).
