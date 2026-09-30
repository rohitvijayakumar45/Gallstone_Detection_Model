# Research Paper Scope — GallStone AI

Literature sweep and gap analysis, 2026-09-27. Every claim about prior work
below links its source. Every claim about this repo traces to a file or an
audit command.

---

## 1. Where the field is

### 1.1 Gallstone detection in ultrasound (direct competitors)

| Work | Data | Split | Result | Calibration / uncertainty / conformal / external |
|---|---|---|---|---|
| Custom DETR + ResNet-50, *CSBJ* Oct 2025 ([PMC12595354](https://pmc.ncbi.nlm.nih.gov/articles/PMC12595354/)) | 1,210 images, 263 patients, 3 Indian hospitals (598 normal / 612 cholelithiasis) | **Image-level** 65/25/10 | mAP@.5:.95 0.32 (DETR) vs 0.30 RT-DETR, 0.24 YOLOv8, 0.18 YOLO-NAS | None of the four. Radiologist κ = 0.78 on test labels |
| Lightweight nets for cholelithiasis/cholecystitis on POCUS, 2021 ([ResearchGate](https://www.researchgate.net/publication/354240310_Lightweight_Deep_Neural_Networks_for_Cholelithiasis_and_Cholecystitis_Detection_by_Point-of-Care_Ultrasound)) | POCUS | — | classification | None |

The closest 2025 paper reports mAP@.5:.95 of 0.32 on an image-level split
with no uncertainty handling. That is the bar, and it is low.

### 1.2 Gallbladder disease / cancer (adjacent, same anatomy)

- **IIT Delhi group** — GBCNet (CVPR 2022), RadFormer (MedIA 2023), FocusMAE
  (CVPR 2024, video, 96.4% acc) ([FocusMAE](https://openaccess.thecvf.com/content/CVPR2024/papers/Basu_FocusMAE_Gallbladder_Cancer_Detection_from_Ultrasound_Videos_with_Focused_Masked_CVPR_2024_paper.pdf)).
  Cancer classification, not stone detection. GBCU dataset is theirs.
- **Multi-centre MIL for GBC**, *Lancet Reg. Health SE Asia* 2026
  ([link](https://www.thelancet.com/journals/lansea/article/PIIS2772-3682(26)00022-3/fulltext)) — multi-centre external validation is now the expected standard for clinical venues.
- **Attention-driven detection for GBC**, *EAAI* 2026 ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0952197626010390)).
- **UIdataGB papers** — many 2025 classification papers report 93–99.85%
  accuracy, e.g. MobResTaNet 99.85% ([arXiv 2512.23033](https://arxiv.org/abs/2512.23033)),
  SE-capsule + ConvBiLSTM ([Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-32978-9)).
  **An audit found near-duplicate frames crossing train/val in UIdataGB; honest
  accuracy fell from 93.05% to ~72.6%** ([uidatagb-xai-reliability](https://github.com/Xrenes/uidatagb-xai-reliability)).

### 1.3 Methods we build on

- **Detector calibration** — pitfalls of D-ECE and fixed thresholds; post-hoc
  temperature scaling as a strong baseline ([Kuzucu et al., ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03148.pdf));
  multivariate calibration ([Küppers 2020](https://arxiv.org/pdf/2004.13546));
  multi-rater calibration for biomedical detection ([2601.23007](https://awesomepapers.io/computer-vision/papers/2601.23007)).
- **Conformal detection** — coordinate-wise CP + Bonferroni, scaled by
  aleatoric uncertainty, tested under cross-domain shift, autonomous driving
  only ([Ries et al. 2026, arXiv 2605.07549](https://arxiv.org/abs/2605.07549), [code](https://github.com/mos-ks/OD-CP));
  sequential risk control ([2505.24038](https://arxiv.org/pdf/2505.24038));
  medical CP is mostly segmentation ([MICCAI 2025 morphological sets](https://arxiv.org/pdf/2503.05618)).
- **Ensemble disagreement as uncertainty** — deep-ensemble disagreement for
  medical anomaly detection ([D2UE, MICCAI 2024](https://papers.miccai.org/miccai-2024/paper/1356_paper.pdf));
  deferral for segmentation ([DeferredSeg 2604.12411](https://arxiv.org/pdf/2604.12411)).
  CNN-vs-transformer disagreement for **detection** in US: not found.
- **Acoustic shadow** — shadow confidence maps ([Meng et al.](https://arxiv.org/pdf/1811.08164)),
  fanlet classification along the beam path at 97% ([SPIE 2022](https://ui.adsabs.harvard.edu/abs/2022SPIE12038E..0MM/abstract)),
  B-mode vs RF statistics ([UMB](https://www.sciencedirect.com/science/article/abs/pii/S0301562919301383)).
  Shadow as a calibration feature for stone detection: not found.
- **Shortcut learning from burned-in text/calipers** in US
  ([Lin et al., MICCAI 2024](https://arxiv.org/pdf/2403.06748)). Relevant:
  Mississippi images carry "GB" text + calipers; our OOD failure carried an arrow.
- **Scanner/vendor domain shift in US** — DG narrows but does not close the gap
  ([JIIM 2026](https://link.springer.com/article/10.1007/s10278-026-02082-z), [DAUS-Net 2026](https://doi.org/10.1177/01617346251388454)).

---

## 2. What this repo actually has (audited)

| Asset | Status |
|---|---|
| 1,593 labelled images, polygon labels | 1,520 are `pseudo_<video>__<frame>` (pseudo-labelled video frames); 73 `pl_*` of unconfirmed provenance |
| Source diversity | **58 video groups** total (30 `a*`, 28 `c*`), median 22 frames/group, max 149 |
| Split | Image-level. **All 49 test groups and 52/52 val groups also appear in train.** Every val/test metric is optimistic |
| Dual detector | YOLOv26L + RF-DETR-L, Optuna-tuned (val mAP 0.96 — leaked split) |
| Calibration | Temperature scaling cut ECE 41% (YOLO) / 36% (RF-DETR) — leaked split |
| Conformal | Box coverage 0.9595 vs 0.95 target — leaked split **and** BGR fusion bug |
| Shadow prior | Hand-coded ratio prior **hurt** F1 (0.93 → 0.83 at τ=0.7; best τ=0.774 still below WBF) |
| OOD observation | External POCUS image: YOLO 0 boxes; RF-DETR only low-confidence boxes (max 0.109) away from the lesion |
| External data on disk | Mississippi abdominal set (Voluson E6, classification labels, 2 confirmed cholelithiasis patients) |
| External data pending | GBCU (1,255 imgs, bbox, needs HOD-signed DUA) |
| Code infra | Calibration, conformal, CRC, selective prediction, XAI, API, UI, regen scripts — all present and tested |

**Consequence:** no current number can go in a paper. The infrastructure can.

---

## 3. Gaps that are real

1. **No gallstone-US detection paper reports calibration, uncertainty, or
   coverage guarantees.** The CSBJ 2025 DETR paper uses none.
2. **Leakage is endemic in this niche** (UIdataGB audit; CSBJ uses image-level
   split). A paper that quantifies the inflation on gallstone *detection* is
   useful by itself.
3. **Conformal detection has not been tested on medical ultrasound or under
   scanner shift.** Ries et al. test cross-domain shift only on driving data.
4. **CNN-vs-transformer disagreement as an OOD / error signal for detection**
   is unexplored in US. We have a concrete motivating case.
5. **Shadow physics has never been used to calibrate stone confidence.** The
   hand-coded version failed; a learned shadow feature fed to multivariate
   calibration is untested.

---

## 4. Paper options (ranked)

### Option A — "Honest gallstone detection" (recommended first paper)

**Title sketch:** *Frame leakage, calibration and conformal coverage in
ultrasound gallstone detection: a reality check.*

Contributions:
1. Quantify metric inflation: image-level vs video-group split on the same
   data and models (mAP, F1, ECE, coverage). Expected: large drop, like UIdataGB.
2. Calibration of YOLO / RF-DETR / fusion under honest splits (ECE, D-ECE,
   reliability).
3. Split-conformal box coverage + CRC recall bound: does coverage survive a
   group split? Does it survive a scanner shift (Voluson / GBCU / POCUS)?
4. Release the split script and leakage checker.

Why it works: needs no new model, uses infrastructure already built, fills
gaps 1–3, and the "reality check" framing is publishable even if numbers are
modest. Venues: MICCAI UNSURE workshop, MIDL short paper,
*Ultrasound in Medicine & Biology*, *Computers in Biology and Medicine*.

### Option B — Cross-architecture disagreement for OOD and deferral

**Hypothesis:** YOLO-vs-RF-DETR disagreement predicts detection errors and
OOD inputs better than either model's confidence.

Measure: AUROC for error detection, risk–coverage curves for selective
prediction, OOD detection on Voluson / GBCU / POCUS images.
Stronger method novelty (gap 4); needs the external sets. Venue: MICCAI main
track or MIDL. Can merge into A as a section if results are thin.

### Option C — Learned shadow prior for calibration

Replace the ratio heuristic with a beam-path (fanlet) shadow score learned
from data; use it as a feature in multivariate calibration rather than a
score multiplier. Honest framing includes the negative result for the
hand-coded prior. Riskiest; do after A.

### Option D — Shortcut robustness (section, not a paper)

Test sensitivity to burned-in text, calipers, arrows (inpaint vs keep) on
Mississippi and POCUS images. Adds credibility to A or B.

---

## 5. Experiments required for Option A

| # | Experiment | Status |
|---|---|---|
| 1 | Re-split by video group; 5-fold grouped CV (58 groups → ~11–12 per fold) | not started |
| 2 | Retrain YOLOv26L + RF-DETR-L per fold with frozen best HPO configs | not started |
| 3 | Swap `map_calculator` per-image F1 for COCO mAP (torchmetrics / pycocotools) | not started |
| 4 | Rerun leaked-split eval with fixed BGR path, as the "inflated" baseline | not started |
| 5 | Calibration: fit on inner fold, report test ECE / D-ECE / Brier; second fold for isotonic | code exists |
| 6 | Conformal: fit on calibration fold, report coverage + width, in-domain and shifted | code exists |
| 7 | External test: hand-box ~150–250 gallstone images from UIdataGB stone class + 2 Mississippi patients + GBCU if granted | not started |
| 8 | Label audit: radiologist review of a pseudo-label sample (Cohen κ vs pseudo labels) | not started — **blocking** |
| 9 | Bootstrap 95% CIs grouped by video | not started |

Compute estimate (RTX-class GPU, from measured run times): RF-DETR-L ≈ 2–3 h
per fold, YOLOv26L ≈ 1 h per fold → 5 folds ≈ **15–20 GPU-hours**. No HPO
rerun needed; reuse best configs from the Optuna DBs.

---

## 6. Open questions to resolve before writing

1. **Who generated the pseudo labels, and how?** 95% of labels are pseudo.
   A reviewer will ask. Need the pipeline and a radiologist-audited sample.
2. **What are the 73 `pl_*` images?** Source and label provenance unknown.
3. **Are `a*` and `c*` two sources/centres?** If so, train-on-`a`/test-on-`c`
   is a free internal cross-site experiment.
4. **Ethics / consent** for the source videos — required for any venue.

---

## 7. Relationship to the patent

Option A strengthens the patent: honest numbers make the claims enabled and
defensible, and the conformal-under-shift result is direct evidence for
claims 1(f), 1(g) and 14. Publishing before filing a provisional would create
prior art against your own claims — **file the provisional first**, then submit.
