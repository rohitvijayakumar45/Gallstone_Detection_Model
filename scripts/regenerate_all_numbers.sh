#!/usr/bin/env bash
# Regenerate every published number in docs/BENCHMARKS.md.
# See docs/REPRODUCIBILITY.md for enablement contract.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 0. Unit tests =="
python -m pytest tests/ -q

echo "== 1. Regenerate MANIFEST =="
python scripts/generate_manifest.py

echo "== 2. Baseline evaluation on test=201 =="
python scripts/comprehensive_evaluation.py \
  --split test \
  --out runs/final_eval/comprehensive_evaluation.json

echo "== 3. Shadow-threshold tuning on val=368 =="
python scripts/tune_shadow_threshold.py \
  --split val \
  --out runs/final_eval/shadow_threshold_tuning.json

echo "== 4. Fit calibration on val=368 =="
python scripts/fit_calibration.py \
  --split val \
  --out-dir production_models \
  --report runs/final_eval/calibration.json

echo "== 5. Fit conformal (calib=val, test=test, alpha=0.05) =="
python scripts/fit_conformal.py \
  --calibration-split val \
  --test-split test \
  --alpha 0.05 \
  --out-dir production_models \
  --report runs/final_eval/conformal.json

echo ""
echo "== DONE =="
echo "Artefacts under production_models/  runs/final_eval/"
echo "Update docs/BENCHMARKS.md by rerunning notebooks/07_ablation_grid.ipynb."
