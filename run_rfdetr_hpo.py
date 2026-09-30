"""
RF-DETR Hyperparameter Optimization — Standalone Script
Runs on dataset_final_resplit (1062 train / 377 val images)
"""
import multiprocessing
multiprocessing.freeze_support()

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc
import torch
from pathlib import Path
import csv
import sys


def objective(trial):
    from src.models.rf_detr_trainer import RFDETRTrainer

    config = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "lr_encoder": trial.suggest_float("lr_encoder", 1e-6, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "clip_max_norm": trial.suggest_float("clip_max_norm", 0.05, 0.5),
    }

    output_dir = f"runs/detect/rfdetr_hpo/rfdetr_trial_{trial.number}"

    trainer_config = {
        "epochs": 50,
        "batch_size": 4,
        "lr": config["lr"],
        "lr_encoder": config["lr_encoder"],
        "weight_decay": config["weight_decay"],
        "clip_max_norm": config["clip_max_norm"],
        "output_dir": output_dir,
        "wandb": False,
        "mlflow": False,
        "patience": 10,
        "num_workers": 2,  # Trying 2 instead of 4 to avoid OOM
        "progress_bar": "tqdm",
    }

    trainer = RFDETRTrainer(trainer_config)
    try:
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}: lr={config['lr']:.6f}, lr_enc={config['lr_encoder']:.7f}, "
              f"wd={config['weight_decay']:.6f}, clip={config['clip_max_norm']:.3f}")
        print(f"{'='*60}")

        model = trainer.train(dataset_dir="dataset_final_resplit", model_size="small")

        # Try to extract mAP50 from model
        val_map50 = 0.0
        if hasattr(model, "metrics") and hasattr(model.metrics, "map50"):
            val_map50 = model.metrics.map50
        elif hasattr(model, "eval_results") and "map50" in model.eval_results:
            val_map50 = model.eval_results["map50"]

        # Fallback: parse metrics.csv written by RF-DETR
        if val_map50 == 0.0:
            metrics_path = Path(output_dir) / "metrics.csv"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        for row in reversed(rows):
                            for key in ["val/mAP_50", "val_map50", "map50", "AP50", "val/map50", "val/mAP_50_95"]:
                                if key in row and row[key]:
                                    val_map50 = float(row[key])
                                    break
                            if val_map50 > 0.0:
                                break

        # Fallback: parse tensorboard events or checkpoint names
        if val_map50 == 0.0:
            # Check for best checkpoint filename which sometimes contains the metric
            best_ckpt = Path(output_dir) / "checkpoint_best_regular.pth"
            if best_ckpt.exists():
                print(f"  Best checkpoint found at {best_ckpt}")
                # If we can't get the metric, report a small value so Optuna doesn't discard
                val_map50 = 0.001

        del model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Trial {trial.number} result: mAP50 = {val_map50:.4f}")
        return val_map50

    except Exception as e:
        print(f"Trial {trial.number} FAILED: {e}")
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        return 0.0


def main():
    print("=" * 60)
    print("RF-DETR Hyperparameter Optimization")
    print(f"Dataset: dataset_final_resplit")
    print(f"Trials: 30 x 50 epochs each")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        study_name="gallstone_rf_detr_hpo",
        storage="sqlite:///rf_detr_hpo_results.db",
        load_if_exists=True,
    )

    # Seed the study with the successful parameters from the first run
    study.enqueue_trial({
        "lr": 5.61e-05,
        "lr_encoder": 7.97e-05,
        "weight_decay": 0.00157,
        "clip_max_norm": 0.319
    })

    study.optimize(objective, n_trials=30, n_jobs=1, show_progress_bar=True)

    print(f"\n{'='*60}")
    print(f"BEST TRIAL: #{study.best_trial.number}")
    print(f"Best mAP50: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
