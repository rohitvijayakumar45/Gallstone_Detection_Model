"""
YOLOv26 Large Hyperparameter Optimization — Standalone Script
Target: ≥97% mAP50 on dataset_final_resplit
20 trials × 50 epochs each, Optuna TPE sampler + MedianPruner

Seeded with the winning yolo26m params as Trial 0.
"""
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc
import torch
from pathlib import Path
import sys
import os
import time

# ── Windows compatibility fixes ──────────────────────────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress MLflow / wandb
os.environ["ULTRALYTICS_MLFLOW"] = "False"
os.environ.pop("MLFLOW_TRACKING_URI", None)

# ── Constants ────────────────────────────────────────────────────────────────
WEIGHTS = "yolo26l.pt"
DATA_YAML = "dataset_final_resplit/data.yaml"
EPOCHS_PER_TRIAL = 50
N_TRIALS = 20
DB_PATH = "sqlite:///yolo26l_hpo_results.db"
STUDY_NAME = "gallstone_yolo26l_hpo"
PROJECT = "runs/detect/yolo26l_hpo"


def objective(trial):
    """Optuna objective: train YOLO26l with sampled hyperparams, return mAP50."""
    try:
        from ultralytics.utils import SETTINGS
        SETTINGS["mlflow"] = False
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Install `ultralytics>=8.3.0` to train YOLO.") from exc

    # ── Sample hyperparameters ───────────────────────────────────────────
    config = {
        "lr0":        trial.suggest_float("lr0", 5e-5, 5e-3, log=True),
        "lrf":        trial.suggest_float("lrf", 0.05, 0.3),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        "box":        trial.suggest_float("box", 4.0, 10.0),
        "cls":        trial.suggest_float("cls", 0.3, 1.2),
        "dfl":        trial.suggest_float("dfl", 1.0, 3.5),
        "cos_lr":     trial.suggest_categorical("cos_lr", [True, False]),
        "mixup":      trial.suggest_float("mixup", 0.0, 0.3),
        "copy_paste": trial.suggest_float("copy_paste", 0.0, 0.3),
        "dropout":    trial.suggest_float("dropout", 0.0, 0.15),
    }

    run_name = f"yolo26l_trial_{trial.number}"

    print(f"\n{'='*70}")
    print(f"  TRIAL {trial.number}")
    print(f"  lr0={config['lr0']:.6f}  lrf={config['lrf']:.4f}  wd={config['weight_decay']:.6f}")
    print(f"  box={config['box']:.3f}  cls={config['cls']:.3f}  dfl={config['dfl']:.3f}")
    print(f"  cos_lr={config['cos_lr']}  mixup={config['mixup']:.3f}  "
          f"copy_paste={config['copy_paste']:.3f}  dropout={config['dropout']:.3f}")
    print(f"{'='*70}")

    t0 = time.time()

    try:
        model = YOLO(WEIGHTS)
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS_PER_TRIAL,
            patience=15,
            imgsz=640,
            batch=6,
            workers=4,
            device=0,
            optimizer="AdamW",
            lr0=config["lr0"],
            lrf=config["lrf"],
            momentum=0.937,
            weight_decay=config["weight_decay"],
            warmup_epochs=5,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=config["box"],
            cls=config["cls"],
            dfl=config["dfl"],
            cos_lr=config["cos_lr"],
            close_mosaic=15,
            amp=True,
            nbs=64,  # effective batch normalization target
            augment=True,
            mixup=config["mixup"],
            copy_paste=config["copy_paste"],
            dropout=config["dropout"],
            mosaic=1.0,
            fliplr=0.5,
            scale=0.5,
            erasing=0.4,
            auto_augment="randaugment",
            project=str(Path(PROJECT).resolve()),
            name=run_name,
            exist_ok=True,
            save=True,
            plots=True,
            verbose=True,
            seed=42,
            deterministic=True,
        )

        # ── Extract mAP50 ───────────────────────────────────────────────
        val_map50 = 0.0

        # Method 1: from model.trainer.metrics (most reliable for ultralytics)
        if hasattr(model, "trainer") and model.trainer is not None:
            metrics = model.trainer.metrics
            if isinstance(metrics, dict):
                val_map50 = metrics.get("metrics/mAP50(B)", 0.0)
            elif hasattr(metrics, "box"):
                val_map50 = getattr(metrics.box, "map50", 0.0)

        # Method 2: from results object
        if val_map50 == 0.0 and results is not None:
            if hasattr(results, "box"):
                val_map50 = getattr(results.box, "map50", 0.0)
            elif hasattr(results, "results_dict"):
                val_map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)

        # Method 3: parse results.csv as fallback
        if val_map50 == 0.0:
            results_csv = Path(PROJECT).resolve() / run_name / "results.csv"
            if results_csv.exists():
                import csv
                with open(results_csv, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Find the best mAP50 across all epochs
                        best_map = 0.0
                        for row in rows:
                            clean = {k.strip(): v.strip() for k, v in row.items()}
                            m = float(clean.get("metrics/mAP50(B)", "0"))
                            if m > best_map:
                                best_map = m
                        val_map50 = best_map

        elapsed = time.time() - t0

        print(f"\n  Trial {trial.number} DONE in {elapsed/60:.1f} min")
        print(f"  >>> mAP50 = {val_map50:.5f} <<<")

        # ── Cleanup ──────────────────────────────────────────────────────
        del model
        if "results" in dir():
            del results
        torch.cuda.empty_cache()
        gc.collect()

        return val_map50

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  Trial {trial.number} FAILED after {elapsed/60:.1f} min: {e}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return 0.0


def main():
    print("=" * 70)
    print("  YOLOv26 LARGE — Hyperparameter Optimization")
    print(f"  Weights:  {WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Trials:   {N_TRIALS} × {EPOCHS_PER_TRIAL} epochs each")
    print(f"  Target:   ≥97% mAP50")
    print(f"  Database: {DB_PATH}")
    print("=" * 70)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=10),
        study_name=STUDY_NAME,
        storage=DB_PATH,
        load_if_exists=True,
    )

    # Seed Trial 0 with winning yolo26m params (strong baseline)
    already_seeded = any(
        t.params.get("lr0") is not None
        and abs(t.params.get("lr0", 0) - 0.000329) < 1e-5
        for t in study.trials
    )
    if not already_seeded:
        study.enqueue_trial({
            "lr0": 0.000329,
            "lrf": 0.1516,
            "weight_decay": 0.00371,
            "box": 5.068,
            "cls": 0.648,
            "dfl": 2.660,
            "cos_lr": False,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "dropout": 0.0,
        })
        print("\n  Seeded Trial 0 with yolo26m winning params")

    remaining = N_TRIALS - len(study.trials)
    if remaining <= 0:
        print(f"\n  Study already has {len(study.trials)} trials. No more needed.")
    else:
        print(f"\n  {len(study.trials)} trials already done. Running {remaining} more...\n")
        study.optimize(objective, n_trials=remaining, n_jobs=1, show_progress_bar=True)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  HPO COMPLETE — {len(study.trials)} total trials")
    print(f"{'='*70}")
    print(f"  BEST TRIAL: #{study.best_trial.number}")
    print(f"  Best mAP50: {study.best_value:.5f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{'='*70}")

    # Print top 5 trials
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)
    print(f"\n  Top 5 Trials:")
    for i, t in enumerate(trials_sorted[:5]):
        print(f"    #{t.number}: mAP50={t.value:.5f}  lr0={t.params.get('lr0', 'N/A')}")


if __name__ == "__main__":
    main()
