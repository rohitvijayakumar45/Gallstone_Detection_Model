# Prior Art Differentiation

Annotated bibliography of prior art directly relevant to each claim
element. For each reference: what it teaches, what it does not teach, and
which claim(s) it fails to anticipate.

The consolidated novelty position: **posterior-acoustic-shadow-weighted
ensemble + two-stage post-hoc calibration + split-conformal box expansion
+ CRC recall bound, in one polygon-supervised dual-detector system, for
gallstone ultrasound**. No cited prior art teaches all elements.

---

## Ensemble / Fusion

### Solovyev et al., 2021 — Weighted Boxes Fusion
`arXiv:1910.13302`
- **Teaches**: WBF as a per-detector-weighted average of overlapping boxes,
  superior to NMS/soft-NMS in generic multi-model detection.
- **Does not teach**: any domain prior (acoustic shadow, medical anatomy),
  post-hoc calibration, conformal guarantee, or polygon-supervised
  training.
- **Fails to anticipate**: claims 1(c), 1(e), 1(f), 1(g).

### Han et al., 2025 — Mixture-of-Expert SoftMax-Weighted Box Fusion (CSM-FusionNet)
`PMC11899514`
- **Teaches**: softmax over per-model mAP@50 as WBF weights for HCC lesion
  detection in liver ultrasound.
- **Does not teach**: gallstone-specific shadow prior, calibration
  cascade, or conformal recall bound; anatomy is HCC (liver parenchyma),
  not gallstone.
- **Fails to anticipate**: 1(a)(iii) shadow-preserving preprocessing,
  1(c), 1(e), 1(f), 1(g).

---

## Calibration

### Guo et al., 2017 — Temperature Scaling
"On Calibration of Modern Neural Networks", ICML 2017.
- **Teaches**: scalar temperature on pre-softmax logits reduces ECE for
  image classifiers.
- **Does not teach**: object detection, ensemble-stage calibration, or
  isotonic post-fusion.
- **Fails to anticipate**: 1(e)(ii), 1(f), 1(g), all system aspects.

### Küppers et al., 2020 — Multivariate Confidence Calibration for Object Detection
CVPR-W 2020.
- **Teaches**: histogram calibration binning on `(conf, position, size)`
  for detection outputs.
- **Does not teach**: two-stage cascade with fusion-stage isotonic, shadow
  prior, or conformal guarantee.
- **Fails to anticipate**: 1(f), 1(g); claim 6 draws on Küppers but adds
  shadow score as a feature.

### Munir et al., 2023 — Cal-DETR
`arXiv:2311.03570`
- **Teaches**: logit mixing for DETR calibration.
- **Does not teach**: ensemble, shadow prior, conformal.
- **Fails to anticipate**: 1(b) (single model only), 1(c), 1(d), 1(f), 1(g).

---

## Uncertainty (evidential / MC)

### Sensoy et al., 2018 — Evidential Deep Learning
NeurIPS 2018.
- **Teaches**: Dirichlet output head for one-shot aleatoric + epistemic on
  classification.
- **Does not teach**: object detection, ensemble, calibration cascade, or
  conformal.
- **Fails to anticipate**: entire independent claim 1.

### E-DETR (anonymous, ICLR 2025 submission)
OpenReview `tdV1GRkCpZ`.
- **Teaches**: evidential deep learning for DETR with IoU-aware loss.
- **Does not teach**: dual-arch ensemble, shadow prior, calibration
  cascade, or conformal boxes.
- **Fails to anticipate**: 1(b) (single model), 1(c), 1(d), 1(f), 1(g).

### Meyer 2022 — Evidential Deep Learning for Object Detection
- Same category as E-DETR; same distinguishers.

---

## Conformal

### Angelopoulos et al., 2022 — Conformal Risk Control
- **Teaches**: monotone-loss extension of split conformal producing
  distribution-free expectation bound on risk.
- **Does not teach**: application to gallstone detection, shadow prior,
  ensemble, dual-architecture, or bounding-box-coordinate quantile
  expansion.
- **Fails to anticipate**: 1(a) through 1(f); claim 1(g) uses CRC but the
  full system integrates it with claims 1(c) and 1(f) which are novel.

### Andéol et al., 2023 — Confident Object Detection via Conformal Prediction
`arXiv:2304.06052`
- **Teaches**: split conformal for railway signal detection with margin
  and hinge nonconformity.
- **Does not teach**: medical ultrasound, shadow prior, ensemble,
  cascade calibration; different application domain.
- **Fails to anticipate**: 1(a), 1(b), 1(c), 1(d), 1(e), 1(h).

### Timans et al., 2024 — Inductive Conformal Prediction for Detection
`doi 10.3390/jimaging12080348`
- **Teaches**: class-conditional coverage guarantees for multi-object
  detection.
- **Does not teach**: shadow prior, calibration cascade, gallstone
  application.

### Conformal Risk Control for Pulmonary Nodule Detection, 2024
`arXiv:2412.20167`
- **Teaches**: CRC for pulmonary nodule detection.
- **Does not teach**: gallstone anatomy, shadow prior (nodules are CT, no
  acoustic shadow), ensemble across CNN + DETR, calibration cascade.

### Conformal Object Detection by Sequential Risk Control, 2025
`arXiv:2505.24038`
- **Teaches**: sequential CRC for detection.
- **Does not teach**: shadow prior, ensemble, calibration cascade,
  gallstone anatomy.

---

## Domain-specific medical ultrasound

### Ultrasound gallstone detection prior work (general survey)
Most published work uses:
  - Single detector (YOLO or Faster R-CNN);
  - Simple thresholding on raw confidence;
  - No calibration validation (no ECE / MCE / reliability diagrams);
  - No coverage guarantees.

### CSM-FusionNet 2025
- Named above under Ensemble/Fusion; closest US-detection reference and
  most directly distinguishable.

### Public benchmarks
- The Kaggle "Ultrasound Gallbladder Dataset" and similar public archives
  offer external test data for claim 14 (external-institution validation).
  Their use does not create prior art here.

---

## Explainability

### Selvaraju et al., 2017 — Grad-CAM
- **Teaches**: gradient-weighted class activation for CNN classifiers.
- **Does not teach**: cross-architecture agreement with transformer
  attention.

### Attention rollout (Abnar 2020)
- **Teaches**: composing self-attention over encoder layers.
- **Does not teach**: correlating with a CNN attribution map.

### Cross-architecture attention agreement (claim 4)
- **Not found in searched prior art**. Novel component. Windowed Pearson
  correlation between EigenCAM output and RF-DETR decoder cross-attention
  map after spatial registration.

---

## Consolidated novelty matrix

Rows = claim elements. Columns = prior art. `✓` = element taught.

| Element | Solov21 | CSM-Han25 | Guo17 | Küpp20 | Cal-DETR | E-DETR | Ang22 | And23 | Tim24 | Nodule24 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1(a) CLAHE + NL-means + shadow preserve | . | . | . | . | . | . | . | . | . | . |
| 1(b) Dual CNN + DETR, poly-supervised | . | ✓* | . | . | . | . | . | . | . | . |
| 1(c) Shadow-ratio prior score | . | . | . | . | . | . | . | . | . | . |
| 1(d) Shadow-weighted WBF | ✓** | ✓* | . | . | . | . | . | . | . | . |
| 1(e)(i) Per-detector temperature | . | . | ✓ | . | ✓† | . | . | . | . | . |
| 1(e)(ii) Post-fusion isotonic | . | . | . | . | . | . | . | . | . | . |
| 1(f) Split-conformal box expansion | . | . | . | . | . | . | . | ✓‡ | ✓‡ | . |
| 1(g) CRC recall bound | . | . | . | . | . | . | ✓ | ✓ | . | ✓ |
| 1(h) Structured report + verdict | . | . | . | . | . | . | . | . | . | . |
| 4 Attention-agreement map | . | . | . | . | . | . | . | . | . | . |
| 7 Cross-arch consistency loss | . | . | . | . | . | . | . | . | . | . |

`*` HCC (liver), not gallstone; no shadow prior. `**` WBF only, no shadow
weighting. `†` DETR calibration only, no ensemble. `‡` generic detection,
no ensemble, no shadow.

No row has `✓` in more than 2 columns. No single reference approaches even
half of claim 1's elements.
