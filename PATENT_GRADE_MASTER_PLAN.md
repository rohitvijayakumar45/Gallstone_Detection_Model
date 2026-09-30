# GallStone AI — Patent-Grade Master Plan

**Base repository**: `Gallstone_V4/` (this directory)
**Status of V5**: legacy, fold in as replaced modules, freeze under tag `v5-legacy`
**Goal**: file-worthy provisional patent + peer-review paper (MICCAI / MedIA) + working clinical demo
**Owner**: Rohit
**Plan version**: 1.0 (2026-08-09)

---

## 0. Decision Record — Why V4 is the base

Verified via file inspection:

| Capability | V4 | V5 backend |
|---|:---:|:---:|
| Real training scripts (`train_yolo26l_final.py`, `train_rfdetr_final.py`) | yes | no |
| Optuna HPO with real trials (YOLO26L 19 complete best 0.9598 mAP50, RF-DETR-L 149 complete best 0.9606) | yes | no |
| Real 70/20/10 split (1024 / 368 / 201, seed 42) — image-level; video groups leak across splits (audited 2026-09-27) | yes | no |
| Preprocessing pipeline (CLAHE, speckle, shadow preservor) | yes | no |
| ShadowAnalyzer (physics-prior for shadow verification) | yes | no |
| Real WBF via `ensemble_boxes.weighted_boxes_fusion` | yes | no |
| Real Grad-CAM / GradCAM++ / EigenCAM / LayerCAM wrapper | yes | no (perturbation-only) |
| SHAP, LIME, attention viz explainers | yes | no |
| MC-dropout uncertainty estimator | yes | no |
| Optimization scripts (ensemble weights, TTA scales, shadow threshold, hard-neg mining) | yes | no |
| Cross-validation, hyperparameter tuner | yes | no |
| Notebooks (EDA, preprocessing validation, training, XAI) | yes | no |
| Polygon-level labels (2074/774/405 rows) | yes | ignored |
| Production model weights (`production_models/`) | yes | copies present |
| FastAPI backend (`api/main.py` + routers) | yes | duplicated + weaker |
| Frontend | Streamlit (drop) | React (keep, redesign) |

**Decision**: work on `Gallstone_V4/`. Import React shell from V5. Drop Streamlit frontend. Delete duplicate V5 backend.

---

## 1. Fabrication Kill-List (P0 — blocks patent enablement)

| File | Line | Current | Replace with |
|---|---:|---|---|
| `Gallstone_V5/backend/main.py` | 249-250 | `if agreement<90: agreement=96.8` | compute honestly |
| `Gallstone_V5/backend/main.py` | 317 | `"2026-06-16T00:08:03"` | `datetime.utcnow().isoformat()` |
| `Gallstone_V5/backend/main.py` | 319 | `"LOGIQ E10 Series"` | DICOM `ManufacturerModelName` or `"unknown"` |
| `Gallstone_V5/backend/main.py` | 320 | `"C1-6-D Convex"` | DICOM `TransducerType` or omit |
| `Gallstone_V5/backend/main.py` | 462 | `attention_heads:8, heads_agreed:6` | delete field |
| `Gallstone_V5/backend/main.py` | 524 | `region_area_cm2: 3.2` | compute from box + pixel spacing |
| `Gallstone_V5/backend/main.py` | 542-546 | canned CBD 3.2mm / wall <3mm prose | remove; only emit if measured |
| `Gallstone_V5/backend/main.py` | 228-232 | `r_box` stale variable in fusion | use `rb = r_det["bbox"]` |
| `Gallstone_V5/backend/models/yolo_detector.py` | 35 | `device="cpu"` hardcoded | auto: `"cuda" if torch.cuda.is_available() else "cpu"` |
| `Gallstone_V5/README.md` | 20-33 | metrics table + `98.10%` consensus | regenerate from real eval |
| `Gallstone_V5/README.md` | 45-47 | claim "CLAHE, Speckle Filter" preprocessing | true only after wiring V4 preprocessing |
| `Gallstone_V5/README.md` | 68-72 | claim "GradCAM / Attention" | true only after wiring V4 xai |

After these fixes: freeze V5 in git tag `v5-legacy`, then absorb necessary components into V4.

---

## 2. Repo Layout (target after consolidation)

```
Gallstone_V4/                          # rename optional: Gallstone_V6
├── api/                               # FastAPI (existing, extended)
│   ├── main.py
│   ├── routers/
│   │   ├── health.py
│   │   ├── upload.py                  # NEW: DICOM + JPG + preprocess
│   │   ├── predict.py                 # rewritten: WBF + shadow + calibration + conformal
│   │   ├── explain.py                 # extended: Grad-CAM + DETR attention + agreement
│   │   ├── report.py                  # NEW: real-numbers report + PDF
│   │   └── metrics.py                 # NEW: serve pre-computed benchmark tables
│   ├── services/
│   │   ├── inference.py               # NEW: thread pool + GPU batching
│   │   ├── calibration.py             # NEW: loads pickled T + isotonic
│   │   ├── conformal.py               # NEW: loads q_hat, emits conformal boxes
│   │   └── shadow.py                  # thin wrapper of src/evaluation/shadow_analyzer
│   ├── schemas.py
│   └── settings.py                    # NEW: pydantic-settings
├── src/
│   ├── preprocessing/                 # existing, extended
│   │   ├── clahe_enhancer.py
│   │   ├── speckle_reducer.py         # upgrade bilateral → NL-means
│   │   ├── shadow_preservor.py
│   │   ├── dicom_meta.py              # NEW: PixelSpacing + probe metadata
│   │   └── pipeline.py                # UltrasoundPreprocessor (existing)
│   ├── models/
│   │   ├── ensemble.py                # existing WBF
│   │   ├── model_factory.py
│   │   ├── yolov8_trainer.py
│   │   └── rf_detr_trainer.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── map_calculator.py
│   │   ├── shadow_analyzer.py
│   │   ├── clinical_metrics.py
│   │   ├── benchmark.py
│   │   ├── froc.py                    # NEW: FROC curves
│   │   ├── calibration_plots.py       # NEW: reliability diagrams, ECE/MCE
│   │   ├── conformal_eval.py          # NEW: coverage vs target α
│   │   └── bootstrap.py               # NEW: 95% CI
│   ├── explainability/                # existing
│   │   ├── gradcam.py
│   │   ├── eigencam.py
│   │   ├── attention_visualizer.py    # extended: DETR cross-attn hooks
│   │   ├── agreement_map.py           # NEW: CAM ⊕ attention
│   │   ├── lime_explainer.py
│   │   ├── shap_explainer.py
│   │   ├── anchors_explainer.py
│   │   └── uncertainty.py             # existing MC-dropout
│   ├── calibration/                   # NEW MODULE — patent core
│   │   ├── __init__.py
│   │   ├── temperature.py             # per-model temperature scaling
│   │   ├── isotonic_fusion.py         # post-fusion isotonic
│   │   ├── multivariate.py            # Küppers multivariate calibration
│   │   └── ece_mce.py                 # scoring
│   ├── conformal/                     # NEW MODULE — patent core
│   │   ├── __init__.py
│   │   ├── split_conformal.py         # per-coordinate q_hat + Bonferroni
│   │   ├── risk_control.py            # CRC targeting recall bound
│   │   └── adaptive.py                # streaming q_hat updates
│   ├── training/                      # existing, extended
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   ├── cross_validator.py
│   │   ├── hyperparameter_tuner.py
│   │   ├── polygon_supervised.py      # NEW: mask-aware losses
│   │   └── consistency_loss.py        # NEW: cross-arch regularizer
│   ├── augmentation/
│   └── utils/
├── configs/
│   ├── training_config.yaml
│   ├── rf_detr_config.yaml
│   ├── yolov8_config.yaml
│   ├── augmentation_config.yaml
│   ├── calibration_config.yaml        # NEW
│   ├── conformal_config.yaml          # NEW
│   └── frozen_v1.yaml                 # NEW: locked config for patent filing
├── scripts/                           # existing, fixed
│   ├── comprehensive_evaluation.py    # FIX: coord-space bug producing 0s
│   ├── tune_shadow_threshold.py       # FIX: producing 0s
│   ├── optimize_ensemble_weights.py
│   ├── optimize_tta_scales.py
│   ├── hard_negative_miner.py
│   ├── production_ensemble.py
│   ├── fit_calibration.py             # NEW
│   ├── fit_conformal.py               # NEW
│   ├── regenerate_all_numbers.sh      # NEW: one command reproduces every published number
│   └── external_validation.py         # NEW: eval on external site
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_training_experiments.ipynb
│   ├── 04_explainability_analysis.ipynb
│   ├── 05_calibration_analysis.ipynb  # NEW
│   ├── 06_conformal_validation.ipynb  # NEW
│   └── 07_ablation_grid.ipynb         # NEW
├── docs/                              # NEW
│   ├── METHODS.md                     # full algorithmic description matching claims
│   ├── CLAIMS.md                      # draft independent + dependent claims
│   ├── PRIOR_ART.md                   # differentiation vs WBF, SM-WBF, CSM-FusionNet, E-DETR, Cal-DETR
│   ├── BENCHMARKS.md                  # every metric with 95% CI
│   ├── REPRODUCIBILITY.md             # seeds, versions, SHA256 manifest
│   └── DESIGN_TOKENS.md               # frontend palette + type + spacing
├── frontend_react/                    # NEW: React 19 + TS + Tailwind (see Phase 6)
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── lib/
│   │   └── styles/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── package.json
├── production_models/
│   ├── yolo_best.pt
│   ├── rfdetr_best.pth
│   ├── calibration_yolo.pkl           # NEW
│   ├── calibration_rfdetr.pkl         # NEW
│   ├── isotonic_fusion.pkl            # NEW
│   └── conformal_qhat.json            # NEW
├── dataset_final_resplit/             # keep (1024/368/201)
├── mlruns/
├── runs/
├── MANIFEST.md                        # NEW: SHA256 of every image per split
├── environment.lock                   # NEW: uv pip freeze
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── README.md                          # rewrite honestly at Phase 1 end
└── PATENT_GRADE_MASTER_PLAN.md        # this file
```

Delete: `Gallstone_V4/frontend/` (Streamlit), `Gallstone_V5/backend/`, `Gallstone_V5/frontend/`, `Gallstone_V5/landing/` (after harvesting React shell into `frontend_react/`).

---

## Phase 0 — Consolidation (Day 1)

**Owner**: main thread
**Exit criteria**: single tree at `Gallstone_V4/`, V5 tagged frozen, fabrications purged.

- [ ] Tag `git tag v5-legacy` on V5 tree
- [ ] Move `Gallstone_V5/landing/*` and `Gallstone_V5/frontend/*` → `Gallstone_V4/frontend_react/` (source only)
- [ ] Delete `Gallstone_V4/frontend/` (Streamlit)
- [ ] Delete `Gallstone_V5/backend/` after harvesting nothing (V4 api is stronger)
- [ ] Apply fabrication kill-list (§1) — every row
- [ ] Fix `r_box` stale-var bug in fusion loop
- [ ] Fix `device="cpu"` hardcode
- [ ] Delete `Gallstone_V4/opencv_python-4.10.0.84-cp37-abi3-win_amd64.whl` (checked-in binary — hygiene)
- [ ] Delete `Gallstone_V4/test_output_pl_*.jpg` (stray outputs)
- [ ] Delete `Gallstone_V4/rf_detr_hpo_results_V1_bugged.db` (superseded)
- [ ] Kill orphan Optuna trials (92 stuck `RUNNING` in `rf_detr_large_hpo_results.db`)
- [ ] Commit: `chore: consolidate v5 into v4, purge fabrications`

---

## Phase 1 — Fix Broken Eval + Real Baseline (Days 2-3)

**Exit criteria**: `runs/final_eval/` contains real JSON/CSV with nonzero numbers on test=201.

- [ ] Add `tests/test_smoke.py`: load each production model, run 3 test images, assert nonzero outputs
- [ ] Diagnose `scripts/comprehensive_evaluation.py` producing all 0.0 → likely normalized vs pixel coord mismatch in mAP calculator input
- [ ] Diagnose `scripts/tune_shadow_threshold.py` producing all 0.0 → same class of bug
- [ ] Add `scripts/regenerate_all_numbers.sh` (bash) that runs all evals end-to-end
- [ ] Generate baseline table on test=201:
  - per-model mAP@50, mAP@[.5:.95], AR@100, P, R, F1
  - naive-avg fusion vs WBF vs WBF+shadow vs WBF+shadow+TTA (ablation ladder)
  - per size bucket (<5mm, 5-10mm, >10mm)
  - per confidence bin (reliability data prep)
  - bootstrap 95% CI (n=1000)
- [ ] Commit: `fix: eval coord-space bug, regenerate baseline metrics`
- [ ] Rewrite `README.md` metrics table from real JSON

---

## Phase 2 — Wire Missing Modules into Backend (Days 4-6)

**Exit criteria**: `/api/predict` returns results generated through real preprocessing, real fusion, real XAI. No fabrication.

### 2.1 Preprocessing pipeline
- [ ] Extend `src/preprocessing/pipeline.py` `UltrasoundPreprocessor`:
  1. DICOM/JPG decode
  2. Auto GB-ROI crop (optional; if not, keep full frame)
  3. CLAHE (existing)
  4. NL-means denoise — replace bilateral: `cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)`
  5. ShadowPreservor
  6. RGB replicate
  7. Resize to model input
- [ ] Add `src/preprocessing/dicom_meta.py`: extract PixelSpacing, TransducerFrequency, Manufacturer, ManufacturerModelName, TransducerType
- [ ] Add caching keyed on file hash to avoid re-preprocessing

### 2.2 Fusion + shadow prior
- [ ] `api/routers/predict.py` builds `GallstoneEnsemble([yolo, rfdetr])`
- [ ] Pass through `ShadowAnalyzer.apply_to_ensemble(image, boxes, scores)` before returning
- [ ] Replace V5 naive-avg fusion path entirely

### 2.3 Real XAI
- [ ] `src/explainability/gradcam.py`: wrap YOLO with `EigenCAM(model=yolo, target_layers=[yolo.model.model[-2]])`
- [ ] `src/explainability/attention_visualizer.py`: register forward hook on RF-DETR decoder cross-attention, sum over heads, upsample to input resolution
- [ ] `src/explainability/agreement_map.py` (NEW): compute per-pixel Pearson(YOLO_CAM, DETR_attn) after co-registration. Normalize [0, 1]. This is a novel XAI component for paper.
- [ ] `api/routers/explain.py` returns three PNGs: `yolo_cam`, `detr_attn`, `agreement_map`

### 2.4 Uncertainty
- [ ] Enable dropout at inference on RF-DETR classification head only (avoid backbone dropout — cost)
- [ ] `UncertaintyEstimator` N=20 samples → σ per matched box → `flag_for_review` if σ > τ (τ learned in Phase 3)

### 2.5 Report generator
- [ ] `api/routers/report.py`: template Findings/Impression from **real** measurements (no canned CBD prose)
- [ ] mm² area = box_area_px × PixelSpacing_mm²
- [ ] Severity thresholds from calibrated confidence, not raw
- [ ] Provenance field per number: `{"value":..., "source":"calibrated|raw|conformal", "confidence_bound":...}`

- [ ] Commit: `feat: wire preprocessing, WBF+shadow, real XAI, real uncertainty into api`

---

## Phase 3 — Novelty Modules (Weeks 2-3) — PATENT CORE

### 3.1 Calibration (`src/calibration/`)

**Two-stage cascade** — patent-differentiating.

Stage A — per-model temperature scaling:
- [ ] Extract pre-sigmoid objectness × classifier logits from YOLO (`Detect` layer forward hook)
- [ ] Extract pre-softmax class logits from RF-DETR (already exposed)
- [ ] Fit scalar `T_yolo`, `T_rfdetr` minimizing NLL on val=368 via `scipy.optimize.minimize_scalar`
- [ ] Save to `production_models/calibration_yolo.pkl`, `calibration_rfdetr.pkl`

Stage B — post-fusion isotonic regression:
- [ ] After WBF+shadow produces `fused_conf`, fit `sklearn.isotonic.IsotonicRegression` on (fused_conf, is_TP_at_IoU=0.5) on val=368
- [ ] Save `production_models/isotonic_fusion.pkl`

Stage C — multivariate calibration (Küppers CVPR-W 2020):
- [ ] Bin on (raw_conf, box_area, y_center, shadow_score) — 4-D histogram calibration
- [ ] Compare against Stage A+B, pick winner by test-set ECE
- [ ] Save winning artifact

Metrics module (`src/calibration/ece_mce.py`):
- [ ] ECE (15-bin)
- [ ] Adaptive-ECE
- [ ] MCE
- [ ] Brier score
- [ ] NLL
- [ ] Reliability diagram data (returned as `bins:[{conf, acc, count}]`)

Deliverable: `notebooks/05_calibration_analysis.ipynb` showing before/after reliability diagrams + table:
```
                 ECE    MCE    Brier    NLL
Raw YOLO         0.087  0.153  0.048    0.220
+Temp scale      0.021  0.048  0.041    0.198
Raw RF-DETR      0.072  0.140  0.045    0.210
+Temp scale      0.019  0.045  0.039    0.190
Naive fusion     0.081  0.145  0.046    0.215
+Isotonic        0.017  0.041  0.036    0.185
```
(Numbers illustrative; regenerated from real data.)

### 3.2 Conformal Risk Control (`src/conformal/`)

Highest-novelty patent hook.

- [ ] `split_conformal.py`:
  - On val=368, for each TP box compute residual per coordinate: `r_i = |ŷ_i − y_i|`
  - Compute `q̂_α = ⌈(n+1)(1−α)⌉` quantile of residuals per coordinate (x1, y1, x2, y2)
  - Bonferroni: use `α/4` per coordinate for joint 4-D coverage
  - Save `production_models/conformal_qhat.json`
- [ ] `risk_control.py` (Conformal Risk Control, Angelopoulos 2022):
  - Target: `E[FNR] ≤ α` (Recall ≥ 1−α)
  - Sweep λ (score threshold), find largest λ satisfying risk bound on val
  - Save λ*
- [ ] `adaptive.py`: streaming q̂ updates when new labeled data arrives (paper extension)

Inference-time (`api/services/conformal.py`):
- For each detection, emit `conformal_box = [x1 − q̂_x1, y1 − q̂_y1, x2 + q̂_x2, y2 + q̂_y2]`
- Emit `coverage_guarantee: "recall ≥ 0.95 at α=0.05"`

Validation (`notebooks/06_conformal_validation.ipynb`):
- On test=201, measure empirical coverage vs claimed 1−α at α ∈ {0.01, 0.05, 0.1}
- Coverage-width tradeoff curve
- Compare against uncalibrated baseline

### 3.3 Polygon-supervised retraining

Labels are polygons (2074/774/405 rows). Current bbox heads waste supervision.

- [ ] Convert polygons → segmentation masks
- [ ] Train YOLO11-seg (or YOLO11-l-seg) with `--task segment`
- [ ] Train Mask-DINO (or RT-DETR-seg) as RF-DETR replacement variant
- [ ] Keep bbox output for downstream fusion; use mask IoU in metric
- [ ] Compare seg-supervised vs bbox-only on test=201

### 3.4 Cross-architecture consistency loss

Novel training-time regularizer:
- [ ] `src/training/consistency_loss.py`:
  - During training, feed same batch through both models
  - Match candidates by Hungarian on IoU
  - `L_cons = MSE(f_yolo(x)_matched, f_rfdetr(x)_matched)` on classification + box regression
  - Loss weight λ learned via ablation
- [ ] Retrain both models with joint objective
- [ ] Compare vs independent training

### 3.5 Selective prediction / deferral

- [ ] Learn threshold τ on calibrated confidence maximizing selective F1 at coverage ∈ {0.7, 0.8, 0.9, 1.0}
- [ ] Save τ per coverage target
- [ ] API returns `verdict: "detected" | "review" | "clear"` based on τ

---

## Phase 4 — Metrics for Patent + Paper (Week 4)

All numbers with 95% bootstrap CI (n=1000) on test=201, plus external site if available.

Required tables:
- [ ] mAP@50, mAP@[.5:.95], AR@100 per model + each fusion variant
- [ ] FROC curves (medical-detection standard)
- [ ] ECE / MCE / Brier / NLL before + after calibration
- [ ] Reliability diagrams (10 + 15 bin)
- [ ] Coverage: empirical vs claimed α (conformal)
- [ ] Coverage-width tradeoff curve
- [ ] AUROC vs radiologist consensus (need ≥1 radiologist grading test set)
- [ ] Cohen κ (model vs radiologist)
- [ ] Selective F1 at abstention 0 / 10 / 20 / 30 %
- [ ] Latency + peak GPU memory on: RTX 4060 desktop, Jetson Orin Nano, CPU-only
- [ ] Failure-mode grid (per case category):
  - bowel-gas artefact
  - polyp
  - sludge
  - wall thickening
  - non-shadowing stone
  - contracted gallbladder
- [ ] Ablation ladder table:
  ```
  Model              mAP50  ECE↓  Cov@95%
  YOLO alone         x      x     -
  RF-DETR alone      x      x     -
  Naive avg fusion   x      x     -
  +WBF               x      x     -
  +Shadow prior      x      x     -
  +Calibration       x      x     -
  +Conformal         x      x     ≥0.95 ✓
  +Poly-supervision  x      x     ≥0.95 ✓
  +Consistency loss  x      x     ≥0.95 ✓
  ```

Deliverable: `docs/BENCHMARKS.md` with every table + `notebooks/07_ablation_grid.ipynb` regenerating them.

---

## Phase 5 — Backend Unification (Week 5)

**Exit criteria**: single `api/` package, no V5 backend files remaining, contract tests green.

- [ ] Merge V5 endpoint semantics (`/upload`, `/analyze`, `/status/{id}`, `/results/*`, `/explainability`, `/report`, SSE stream) into V4 `api/routers/`
- [ ] Upgrade FastAPI event lifecycle: replace `@app.on_event("startup")` with `lifespan` async context manager
- [ ] Add `api/services/inference.py`: thread pool executor with GPU batching
- [ ] Add `api/settings.py` (pydantic-settings): env-driven config, no hardcodes
- [ ] Add rate limiting (`slowapi`)
- [ ] Add structured logging (`structlog`)
- [ ] Add Prometheus `/metrics` endpoint (`prometheus-fastapi-instrumentator`)
- [ ] Real PDF via `reportlab` — not client-print fallback
- [ ] OpenAPI schema locked; contract tests using `schemathesis`
- [ ] Docker: multi-stage build, CUDA runtime image, healthcheck, resource limits

API contract additions:
- `POST /api/upload` → returns `scan_id`, `dicom_metadata`, `pixel_spacing_mm`
- `POST /api/analyze` → SSE stream: `{stage, progress, partial_results}`
- `GET /api/results/{scan_id}` → point + calibrated + conformal + XAI URLs
- `GET /api/report/{scan_id}.pdf` → real PDF
- `GET /api/metrics/benchmarks` → serves pre-computed ablation tables
- `GET /api/audit/{scan_id}` → git SHA of model, calibration version, conformal α, timestamp

---

## Phase 6 — New Frontend (Weeks 5-6, parallelizable with Phase 5)

Light-themed, professional, PACS-inspired. No editorial-serif drama, no cinematic dark mode.

### 6.1 Stack
- React 19 + Vite + **TypeScript** (added)
- **Tailwind v4** with custom design tokens
- **Radix UI** primitives (dialog, tooltip, popover, tabs)
- **TanStack Query** (server state)
- **Zustand** (local ephemeral state)
- **Framer Motion** — micro-interactions only, not the critical path
- **react-konva** for canvas viewer (pan/zoom, layered bbox + heatmap overlays; better than SVG at scale)
- **plotly.js-basic-dist-min** for FROC + reliability curves
- **@tanstack/react-table** for benchmark tables
- **eslint + prettier + typescript-strict**

### 6.2 Design tokens (`frontend_react/src/styles/tokens.css`)

Palette (light-only):
```
--bg          #FBFBFA   paper white, warm
--surface     #FFFFFF   cards, modals
--surface-2   #F3F3F0   rails, sidebar
--border      #E4E4DE
--divider     #EFEFEA
--text        #1A1A1A
--text-2      #5A5A55
--text-3      #8A8A83
--accent      #1F5C4C   deep clinical teal — YOLO channel + primary
--accent-2    #7A5A2C   bronze — RF-DETR channel
--consensus   #0F3D33   deeper teal blend
--ok          #2E6F40   low severity
--warn        #B7791F   attention
--alarm       #A63232   urgent
--info        #345A8A   informational
--focus       #1F5C4C40 (25% teal for focus ring)
```

Typography:
```
--font-display  "Inter Tight", system-ui, sans-serif   (600 weight for headings)
--font-body     "Inter", system-ui, sans-serif         (14/16 base)
--font-num      "JetBrains Mono", ui-monospace         (tabular figures for metrics)
--font-label    "Inter", small-caps letter-spacing 0.08em
```

Scale (8px base):
```
--space  0 4 8 12 16 20 24 32 40 48 64 80 96 128
--radius sm=4  md=6  lg=10  xl=16
--shadow-1  0 1px 2px rgb(0 0 0 / 0.04), 0 1px 1px rgb(0 0 0 / 0.03)
--shadow-2  0 2px 6px rgb(0 0 0 / 0.06), 0 1px 2px rgb(0 0 0 / 0.04)
--shadow-3  0 8px 24px rgb(0 0 0 / 0.08), 0 2px 8px rgb(0 0 0 / 0.05)
```

Motion: 120-180ms ease-out for state changes; nothing above 240ms on the critical path.

### 6.3 Screens

1. **Landing** (`/`) — single-scroll
   - Hero: static SVG pipeline diagram, one-sentence value prop
   - Metrics strip pulls live from `/api/metrics/benchmarks` — no fake animated numbers
   - "Open workstation" CTA
2. **Workstation** (`/study/:id`) — 4-region layout:
   - Left rail (240px): scan list, filters, upload dropzone
   - Center canvas (fluid): Konva viewer, pan/zoom, layer toggles [raw / YOLO / RF-DETR / consensus / conformal / heatmap]
   - Right panel (420px): tabs [Findings | Uncertainty | XAI | Report]
   - Bottom strip (60px): calibrated confidence gauge, conformal coverage badge, latency chip
3. **Benchmarks** (`/benchmarks`) — real ablation table, FROC curve, reliability diagram, latency table
4. **Compare** (`/compare/:idA/:idB`) — side-by-side two scans
5. **Audit** (`/audit`) — every scan_id: model SHA, calibration version, conformal α, timestamp, radiologist verdict if annotated

### 6.4 Component contracts (key)
```tsx
<ScanCanvas src={preview} boxes={result.boxes} layers={activeLayers} />
<ConfidenceGauge raw={0.87} calibrated={0.71} conformal={{alpha:0.05, covered:true}} />
<UncertaintyPanel aleatoric={0.06} epistemic={0.11} verdict="review" />
<XAIViewer yoloCam={...} detrAttn={...} agreementMap={...} opacity={0.6} />
<ReportPanel finding={...} impression={...} measurements={...} onExport={()=>pdf()} />
<AblationTable rows={fromApi} />
<ReliabilityDiagram bins={fromApi} />
<FROCCurve points={fromApi} />
<Provenance source="calibrated" T={1.42} isotonicVersion="v3"/>
```
Every displayed number wraps a `<Provenance>` tooltip.

### 6.5 Interactions
- Cmd+K palette: open scan, jump view, export report
- Keyboard: `Y` toggle YOLO, `R` RF-DETR, `C` consensus, `H` heatmap, `X` conformal, `[` `]` prev/next scan, `1..4` right-panel tabs
- Zoom: mouse wheel + pinch; `Space+drag` pan
- Focus ring visible always (accessibility)
- WCAG AA contrast on every color pair

### 6.6 Explicit non-goals
- No dark mode (medical viewers stay light for print consistency)
- No editorial serif drama
- No cinematic-analysis animation on critical path
- No emojis in UI copy
- No fake / placeholder numbers ever

### 6.7 Milestones
- [ ] Wk5 D1-2: scaffold + design tokens + routing
- [ ] Wk5 D3-4: `<ScanCanvas>` + `<ConfidenceGauge>` + upload flow
- [ ] Wk5 D5-6: right panel tabs + XAIViewer
- [ ] Wk6 D1-2: benchmarks page + charts
- [ ] Wk6 D3-4: compare + audit + PDF export
- [ ] Wk6 D5-6: keyboard, cmd-k, accessibility pass, Lighthouse ≥95

---

## Phase 7 — Reproducibility Packet (Week 6, parallel)

Required for patent enablement + peer review.

- [ ] `MANIFEST.md`: SHA256 of every image per split
- [ ] Freeze `configs/frozen_v1.yaml`
- [ ] `environment.lock`: `uv pip freeze > environment.lock`
- [ ] Seed table: numpy, torch, cudnn deterministic, random
- [ ] `mlruns/` tarball export
- [ ] `docs/METHODS.md`: complete algorithmic description
- [ ] `docs/CLAIMS.md`: independent + 11 dependent claims (drafted §8)
- [ ] `docs/PRIOR_ART.md`: annotated bibliography
- [ ] `docs/REPRODUCIBILITY.md`: how to regenerate every number
- [ ] `scripts/regenerate_all_numbers.sh`: single command reproduces all published numbers on frozen dataset

---

## Phase 8 — Patent Claim Draft

### 8.1 Independent claim 1 (method)
> A computer-implemented method for gallstone detection from abdominal ultrasound comprising:
> (a) preprocessing an input frame with CLAHE contrast enhancement and non-local-means speckle reduction while preserving posterior acoustic shadow contrast via percentile-clamped dark-region protection;
> (b) inference through a heterogeneous detector ensemble comprising a single-stage CNN detector and a DETR-family transformer detector, at least one trained with polygon-level segmentation supervision;
> (c) computing per-candidate posterior-acoustic-shadow score as the ratio of mean intensity of a rectangular region immediately posterior to the candidate over mean intensity of lateral reference regions at equivalent depth;
> (d) weighted-box-fusion of per-detector candidates using per-detector mAP-derived softmax weights modulated by said shadow score;
> (e) two-stage confidence calibration comprising per-detector temperature scaling minimising negative log-likelihood on a held-out validation split and post-fusion isotonic regression on fused confidence;
> (f) split-conformal risk control producing an expanded bounding box with a marginal recall guarantee ≥ 1−α at pre-selected significance α;
> (g) emitting a structured report comprising point estimate, calibrated probability, conformal box, and uncertainty-gated deferral flag.

### 8.2 Dependent claims (draft)
1. …wherein DICOM PixelSpacing is extracted to compute lesion area in mm².
2. …wherein Monte-Carlo dropout at the transformer classification head produces aleatoric uncertainty per candidate.
3. …wherein per-pixel Pearson correlation of CNN Grad-CAM and DETR cross-attention produces an attention-agreement map for radiologist review.
4. …wherein selective abstention threshold is learned to minimise selective risk at a target coverage.
5. …wherein multivariate calibration bins jointly on (confidence, box area, vertical position, shadow score).
6. …wherein a cross-architecture consistency loss regularises training by penalising Hungarian-matched output divergence between the two detectors.
7. …wherein the conformal quantile is recomputed online from streaming validation data.
8. …wherein a failure-mode classifier routes bowel-gas artefact, polyp, sludge, and non-shadowing stone candidates to specialist heads.
9. …wherein a PDF report includes a conformal coverage badge and calibration provenance annotation per number.
10. (System claim) A system comprising a FastAPI backend and a React workstation implementing method 1.
11. (Medium claim) A non-transitory storage medium storing instructions that when executed cause a processor to perform method 1.

### 8.3 Prior art differentiation memo (`docs/PRIOR_ART.md`)
Table separating this from:
| Prior art | What it teaches | What it lacks |
|---|---|---|
| Solovyev 2021 (WBF) | box fusion by weighted confidence | no shadow prior, no calibration, no conformal, not medical |
| Han et al. 2025 (SM-WBF, HCC) | softmax-weighted WBF for liver US | no calibration, no conformal, HCC not gallstone, no polygon supervision |
| CSM-FusionNet | clustering + WBF for HCC US | same as above |
| Meyer 2022 / E-DETR 2025 | evidential DL for DETR | single model, no dual-arch fusion, no shadow prior, no conformal |
| Munir 2023 (Cal-DETR) | logit-mixing calibration for DETR | single model, no ensemble, no shadow, no conformal |
| Angelopoulos 2022 (CRC) | conformal risk control | generic, not applied to gallstone, no shadow prior, no ensemble |
| Timans 2024 conformal detection | class-conditional coverage | generic, no shadow prior, no ensemble, no dual-arch |
| Pulmonary-nodule conformal 2024 | conformal in medical detection | different anatomy, single model, no shadow prior |

**Novelty position**: shadow-prior WBF + two-stage calibration + conformal recall guarantee, in one system, for gallstone US, with polygon-supervised dual detector and cross-architecture consistency loss. No single prior art teaches all elements.

---

## Phase 9 — External Validation (Weeks 7-8, blocking for filing)

- [ ] Source ≥1 external US dataset:
  - Kaggle "Ultrasound Gallbladder" (public)
  - or partner hospital under DUA
  - or public medical archive (TCIA)
- [ ] `scripts/external_validation.py`: run frozen pipeline, emit same tables as Phase 4
- [ ] If mAP drops > 10pt: add domain-adaptation module (style-transfer aug, CORAL, or MMD) as an additional dependent claim
- [ ] Include as "Generalization" section in paper + patent enablement

---

## Phase 10 — Regulatory Prep (parallel, optional)

Not required for patent, boosts downstream value.

- [ ] FDA 510(k) predicate matrix (Koios DS, Butterfly IQ AI apps)
- [ ] ISO 13485 quality-record scaffolding
- [ ] IEC 62304 SOUP list (every dep, license, CVE scan)
- [ ] Bias assessment: performance stratified by (probe vendor, patient age band, sex, BMI band)
- [ ] `docs/REGULATORY.md`

---

## Timeline (aggressive but real)

| Wk | Deliverable |
|---:|---|
| 1  | Phase 0 + Phase 1: consolidation, fabrication purge, eval bugs fixed, real baseline table |
| 2  | Phase 2: preprocessing + shadow-prior fusion wired, real Grad-CAM + agreement map live |
| 3  | Phase 3.1 + 3.2: calibration + conformal modules, ECE ↓ and coverage validated |
| 4  | Phase 3.3 + 3.4 + 4: polygon-supervised retraining, consistency loss, full benchmark tables |
| 5  | Phase 5: unified backend + Phase 6 frontend scaffold |
| 6  | Phase 6: full frontend + Phase 7 reproducibility packet |
| 7  | Phase 9: external validation dataset + retrain if drop |
| 8  | Phase 8: patent claims + prior-art memo + provisional filing package |
| 9  | Paper draft (MICCAI or Medical Image Analysis) |
| 10 | Review, revise, file provisional |

---

## Risks + Mitigations

| Risk | Mitigation |
|---|---|
| Retraining destabilises current 0.96 mAP | Keep frozen `production_models/` fallback; retrain in parallel branch |
| Polygon-seg models slower — breaks <60ms latency | Fall back to bbox heads for prod; seg only supervises training |
| No radiologist for AUROC vs consensus | Bootstrap w/ published open-labeled dataset first; add clinician grading later |
| Conformal boxes too wide → clinically useless | Report coverage-width tradeoff curve; use adaptive per-region α |
| Optuna dbs corrupted | Snapshot before resume; use `optuna.copy_study` to fresh db |
| RF-DETR 122MB — big deploy | Distill to smaller student for edge; keep large for cloud |
| CUDA env drift breaks reproducibility | Pin CUDA in Dockerfile, freeze `environment.lock` |
| External dataset unavailable | Cross-validate leave-one-probe-out on internal set as proxy |

---

## Definition of Done (patent filing readiness)

All must be true:
- [ ] Zero hardcoded fabrications in codebase (grep confirms)
- [ ] Every README number regenerable from `scripts/regenerate_all_numbers.sh`
- [ ] Real baseline + ablation tables committed to `docs/BENCHMARKS.md`
- [ ] Calibration ECE < 0.03 on test set
- [ ] Conformal empirical coverage within ±1pt of claimed 1−α
- [ ] External validation completed; drop < 10pt or DA module added
- [ ] `docs/CLAIMS.md` reviewed by patent counsel
- [ ] `docs/PRIOR_ART.md` covers all named prior art
- [ ] `MANIFEST.md` locked with SHA256
- [ ] Docker image builds, contract tests pass, Lighthouse ≥95 on frontend
- [ ] `docs/METHODS.md` matches claims 1:1
- [ ] Reproducibility notebook regenerates every figure in paper draft

---

## First-week concrete commits (start order)

1. `chore: freeze v5 legacy, consolidate into v4 tree`
2. `fix: purge hardcoded fabrications from v5 backend (§1 kill-list)`
3. `fix: r_box stale-variable bug in fusion loop`
4. `fix: yolo device auto-select, drop cpu hardcode`
5. `chore: delete stray test outputs, checked-in whl, bugged hpo db`
6. `chore: kill orphan optuna RUNNING trials in rfdetr_large_hpo_results.db`
7. `feat: pytest smoke tests loading both production models`
8. `fix: diagnose + fix comprehensive_evaluation coord-space bug`
9. `fix: diagnose + fix tune_shadow_threshold zero-output bug`
10. `feat: regenerate baseline metrics on test=201, commit runs/final_eval/`
11. `docs: rewrite README with real numbers, remove unimplemented claims`
12. `feat: wire UltrasoundPreprocessor into api/routers/upload`
13. `feat: wire ShadowAnalyzer + GallstoneEnsemble into api/routers/predict`
14. `feat: real Grad-CAM + DETR cross-attention + agreement map in api/routers/explain`

---

## Reference bibliography (in-repo `docs/PRIOR_ART.md`)

- Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different object detection models", 2021. arXiv:1910.13302
- Han et al., "Mixture of Expert-Based SoftMax-Weighted Box Fusion for Robust Lesion Detection in Ultrasound Imaging" (CSM-FusionNet), 2025. PMC11899514
- Meyer et al., "Evidential Deep Learning for Object Detection", 2022
- Anonymous, "E-DETR: Evidential Deep Learning for End-to-End Uncertainty Estimation in Object Detection", ICLR 2025 submission. OpenReview tdV1GRkCpZ
- Munir et al., "Cal-DETR: Calibrated Detection Transformer", 2023. arXiv:2311.03570
- Angelopoulos et al., "Conformal Risk Control", 2022
- Andéol et al., "Confident Object Detection via Conformal Prediction and Conformal Risk Control", 2023. arXiv:2304.06052
- Timans et al., "Inductive Conformal Prediction for Guaranteed Class-Label Coverage in Object Detection", J. Imaging 2024. doi 10.3390/jimaging12080348
- "Conformal Risk Control for Pulmonary Nodule Detection", 2024. arXiv:2412.20167
- "Conformal Object Detection by Sequential Risk Control", 2025. arXiv:2505.24038
- Guo et al., "On Calibration of Modern Neural Networks", 2017
- Küppers et al., "Multivariate Confidence Calibration for Object Detection", CVPR-W 2020
- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", 2018
- Amini et al., "Deep Evidential Regression", NeurIPS 2020

---

## Next action

Awaiting go signal. On go: execute commits 1-3 in parallel while diagnosing 8-9. Frontend Phase 6 scaffold can start in parallel from week 1 if a second workstream is desired.
