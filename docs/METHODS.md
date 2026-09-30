# METHODS

Complete algorithmic description of the GallStone AI v6 pipeline. Every
element here maps 1:1 to a claim in [CLAIMS.md](CLAIMS.md); each claim's
enablement lives in this file. Reviewers should be able to reimplement the
system from this document alone, then reproduce every published number via
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## 1. Notation

Symbols used throughout:

- `I` — input frame (grayscale ultrasound; RGB by triplication for CNN input)
- `w, h` — original image width, height in pixels
- `b = (x1, y1, x2, y2)` — axis-aligned bounding box in pixel space
- `b̃` — the same box in normalized image space `[0, 1]^4`
- `s ∈ [0, 1]` — raw detector confidence
- `sT` — confidence after per-detector temperature scaling
- `sf` — fused confidence after weighted-box-fusion + shadow prior
- `s*` — post-fusion isotonic-calibrated probability
- `α` — target miscoverage (default `0.05`)
- `q̂ = (q̂_{x1}, q̂_{y1}, q̂_{x2}, q̂_{y2})` — split-conformal per-coordinate quantiles
- `λ*` — Conformal-Risk-Control score threshold

---

## 2. Preprocessing (`src/preprocessing/pipeline.py`)

Given `I` of arbitrary size, the pipeline applies:

1. **Grayscale conversion** if `I` is RGB (`cv2.cvtColor(BGR→GRAY)`).
2. **Ultrasound-region crop** — Otsu-style intensity threshold at 10;
   contour-of-largest-area with 5-pixel margin. Skipped if the resulting
   region is under 5 % of frame area (transducer overlay dominant).
3. **Speckle denoising** — configurable:
   - `bilateral` (legacy, `cv2.bilateralFilter` `d=9, σc=σs=75`);
   - `nlmeans` (default for training-time reproducibility, `cv2.fastNlMeansDenoising`
     `h=10, template=7, search=21`).
4. **CLAHE** — `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`.
5. **Auto-gamma** — `γ = 0.7` if mean < 0.3; `γ = 1.3` if mean > 0.7; else `1.0`.
6. **Percentile normalization** — clip to `(p1, p99)` and rescale to `[0, 255]`.
7. **Shadow-preserving clamp** — `ShadowPreservor` re-injects dark regions
   (`I < p20`) so posterior acoustic shadows are not washed out by CLAHE.
8. **Letterbox resize** to model input side (default 640) with aspect ratio
   preserved; pad recorded as `(pad_w, pad_h)` and geometric ratio `r` so
   downstream coordinates can be inverted.

DICOM ingest additionally extracts `PixelSpacing`, `ManufacturerModelName`,
`TransducerType`, `TransducerFrequency`, `StudyInstanceUID`, `PatientAge`,
`PatientSex`, `BodyPartExamined` (see `src/preprocessing/dicom_meta.py`).
`PixelSpacing` is the only field required for real-unit measurements; all
others are provenance.

---

## 3. Detector Ensemble (`src/models/ensemble.py`)

Two heterogeneous detectors run in parallel on the preprocessed frame:

- **YOLOv26L** — single-stage CNN, ~24.7 M parameters. Trained with `ultralytics`
  under Optuna hyperparameter search (`run_yolo26l_hpo.py`, best val mAP@50 =
  0.9598 across 19 completed trials, see `yolo26l_hpo_results.db`).
- **RF-DETR-L** — DETR-family transformer with DINOv2 backbone. Trained with
  the `rfdetr` package under Optuna (`run_rfdetr_large_hpo.py`, best val
  mAP@50 = 0.9606 across 140 completed trials, see
  `rf_detr_large_hpo_results.db`).

Both operate on the same preprocessed frame at 640×640. Boxes are converted
to normalized `[0, 1]^4` space for fusion.

### 3.1 Weighted Box Fusion

The fusion function is Solovyev et al.'s WBF (`ensemble_boxes` package,
Solovyev 2021), invoked with per-model weights `w = (0.5, 0.5)` by default
or optimized via `scripts/optimize_ensemble_weights.py`. WBF differs from
NMS/soft-NMS by *averaging* overlapping boxes weighted by confidence rather
than discarding them.

### 3.2 Posterior-Acoustic-Shadow Prior (`src/evaluation/shadow_analyzer.py`)

Gallstones' defining sonographic sign is a *posterior acoustic shadow* — a
dark cone extending inferior to the calculus because sound is fully
reflected. We encode this as a physics prior on top of WBF.

For each fused candidate `b = (x1, y1, x2, y2)` with score `sf`:

1. Define **shadow ROI** as the rectangle `(x1, y2, x2, y2 + D)` clipped to
   image, with `D = 100 px` by default.
2. Define **lateral reference ROIs** as `(x1 - m, y2, x1, y2 + D)` and
   `(x2, y2, x2 + m, y2 + D)` where `m = (x2 - x1) / 2`.
3. Compute `ratio = mean(shadow_ROI) / mean(lateral_ROIs)`.
4. Modulate confidence:
   - `ratio < threshold`: strong shadow — multiply by `1.2` (clipped to 1.0).
   - `ratio > 0.95`: no shadow — multiply by `0.5` (likely gas artefact).
   - otherwise: leave unchanged.

The threshold is grid-tuned by `scripts/tune_shadow_threshold.py` on the
val split (368 images) using per-image mean-F1 at IoU 0.5.

### 3.3 Test-Time Augmentation (optional)

`GallstoneEnsemble.predict_multi_scale` runs both detectors at scales
`{640, 800, 1024}` (tunable via `scripts/optimize_tta_scales.py`), fusing
via WBF with per-scale weights.

---

## 4. Calibration (`src/calibration/`)

Raw detector confidences are not proper probabilities. We calibrate in two
stages so the fused output can be interpreted as `P(TP | detection)`.

### 4.1 Per-Detector Temperature Scaling (`temperature.py`)

Given held-out `(s_i, y_i)` pairs on val (368 images, IoU-matched at 0.5):

`sT_i = σ(logit(s_i) / T)` where `logit(s) = log(s / (1 - s))`.

Fit `T ∈ (0.05, 20)` minimizing binary cross-entropy via `scipy`'s bounded
scalar optimizer. Save `T` per detector (`calibration_yolo.pkl`,
`calibration_rfdetr.pkl`).

### 4.2 Post-Fusion Isotonic Regression (`isotonic_fusion.py`)

After WBF + shadow, fit `sklearn.isotonic.IsotonicRegression` on
`(fused_conf, is_TP)` — same val split, fresh fold recommended. Save to
`isotonic_fusion.pkl`. Monotone by construction; output clipped to `[0, 1]`.

### 4.3 Multivariate Calibration (optional; `multivariate.py`)

4-D quantile-bin histogram calibrator on `(conf, box_area, y_center,
shadow_score)`. Per-bin empirical `P(TP)` with Laplace smoothing
(pseudocount 1). Used when the univariate isotonic is not sufficient; the
production stack currently uses temperature + isotonic and reserves
multivariate for the paper's ablation.

### 4.4 Scoring (`ece_mce.py`)

We report ECE (15 equal-width bins), adaptive-ECE (15 equal-mass quantile
bins), MCE, Brier score, NLL, and 10-bin reliability data for reliability
diagrams (see `notebooks/05_calibration_analysis.ipynb`).

### 4.5 Selective Prediction (`selective.py`)

Learn `τ` per target coverage `c ∈ {0.7, 0.8, 0.9, 1.0}` maximizing F1 on
the covered subset. Persisted in `selective_thresholds.json`. At inference,
`verdict = "detected"` if `s* ≥ τ`; `"review"` if `τ/2 ≤ s* < τ`; else
`"clear"`.

---

## 5. Conformal Risk Control (`src/conformal/`)

Provides distribution-free coverage guarantees on the emitted boxes and on
the recall of the detection set.

### 5.1 Split-Conformal Box Expansion (`split_conformal.py`)

For each matched calibration pair `(p, g)` in normalized coordinates,
compute per-coordinate residuals:

```
r_x1 = max(0, p.x1 - g.x1)
r_y1 = max(0, p.y1 - g.y1)
r_x2 = max(0, g.x2 - p.x2)
r_y2 = max(0, g.y2 - p.y2)
```

With Bonferroni over 4 coordinates (`α' = α/4`), take the small-sample
conformal quantile `q̂_c = sort(r_c)[⌈(n+1)(1 - α')⌉ - 1]` for each `c`.

At inference, expand a new prediction `p` to `p_conf`:

```
p_conf = (p.x1 - q̂_x1, p.y1 - q̂_y1, p.x2 + q̂_x2, p.y2 + q̂_y2)
```

clipped to the image. Marginal guarantee: `P(g ⊂ p_conf) ≥ 1 - α`.

### 5.2 Conformal Risk Control for Recall (`risk_control.py`)

Following Angelopoulos 2022, choose the largest `λ ∈ [0, 1]` such that

```
(n / (n + 1)) · mean_FNR(λ) + 1 / (n + 1) ≤ α
```

where `FNR(λ)` on a calibration image is the fraction of GT boxes not
matched (IoU ≥ 0.5) by any prediction with score `≥ λ`. This gives
`E[FNR] ≤ α`, i.e., expected recall `≥ 1 - α`.

### 5.3 Adaptive / Streaming (`adaptive.py`)

`AdaptiveConformalBox` keeps a bounded FIFO of residuals and re-derives `q̂`
on demand. Enables online recalibration when radiologist verdicts stream
back post-deployment.

---

## 6. Explainability (`src/explainability/`)

Three-channel XAI, all served by `/api/explain/{scan_id}`:

- **YOLO EigenCAM** — `pytorch_grad_cam.EigenCAM` on `model.model[-2]`
  (feature layer immediately before the Detect head). Renders as JET overlay.
- **RF-DETR cross-attention** — `AttentionVisualizer` walks the module tree
  to find decoder layers, hooks each `cross_attn` forward, averages weights
  over heads and object queries in the final decoder layer, reshapes to
  spatial grid, and renders as JET overlay.
- **Attention-agreement map** — per-pixel windowed Pearson correlation
  (kernel 17×17) between the two normalized maps. Novel contribution: high
  agreement = cross-architectural corroboration; low agreement = only one
  architecture "trusts" the region. Rendered as Turbo overlay.

Additionally, `src/explainability/uncertainty.py` provides Monte-Carlo
dropout at the RF-DETR classification head (`N = 20` forward passes with
`model.train()`) — aleatoric proxy via σ of per-pass confidences with an
in-built `flag_for_review` gate at `σ > 0.15`.

---

## 7. Report Generation (`api/routers/report.py`)

Only fields backed by real measurements appear in the report:

- **Finding** — `"Gallstone(s) detected"` or `"No gallstone detected"`.
- **Impression** — calibrated probability + conformal recall guarantee (if
  loaded) + clinician-review disclaimer.
- **Measurements**: per detection: calibrated confidence, area in pixels,
  area in mm² *iff* DICOM `PixelSpacing` was extracted, conformal area,
  verdict (`detected` / `review` / `clear`).
- **Provenance**: calibration version, conformal α, pixel spacing source,
  device, probe, ISO-format `generated_at`.

No canned prose about wall thickness, common bile duct, or adjacent
structures is emitted; those must be independently reported by the
clinician.

---

## 8. Runtime

- Backend: FastAPI + lifespan-managed model warmup, thread-pool inference
  (`api/services/inference.py`), pydantic-settings-driven configuration
  (`api/settings.py`), pre-computed metrics served by `/api/metrics/*`.
- Frontend: React 19 + TypeScript + Vite + Tailwind. Light-only, no dark
  mode. Konva canvas for pan/zoom/layers. Every numeric field renders a
  `<Provenance>` tooltip indicating source (`raw` / `calibrated` /
  `conformal` / `selective`).

---

## 9. Reproducibility

- Data: `dataset_final_resplit/` — 1024 train / 368 val / 201 test, patient-
  level split, seed 42, source manifest in
  `dataset_final_resplit/split_info.json`. Per-image SHA256 in `MANIFEST.md`.
- Weights: `production_models/yolo_best.pt`, `rfdetr_best.pth`.
- Calibration + conformal artefacts: fit by `scripts/fit_calibration.py` +
  `scripts/fit_conformal.py`; frozen into `production_models/`.
- Regenerate every published number: `scripts/regenerate_all_numbers.sh`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact commands, expected
runtime, and environment lockfile.
