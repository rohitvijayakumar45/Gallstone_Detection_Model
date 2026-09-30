import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gc
import torch
from pathlib import Path

class HyperparameterTuner:
    def __init__(self, model_type="yolo", n_trials=30):
        self.model_type = model_type
        self.n_trials = n_trials

    def objective(self, trial):
        if self.model_type == "yolo":
            config = {
                "lr0": trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
                "lrf": trial.suggest_float("lrf", 0.01, 0.2),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                "box": trial.suggest_float("box", 5.0, 10.0),
                "cls": trial.suggest_float("cls", 0.3, 1.0),
                "dfl": trial.suggest_float("dfl", 1.0, 3.0),
                "optimizer": trial.suggest_categorical("optimizer", ["SGD", "AdamW", "auto"]),
                "epochs": 50, # extended training for better HPO
                "name": f"hpo_trial_{trial.number}"
            }
            val_map50, val_recall = self._train_and_evaluate(config)
            return 0.7 * val_map50 + 0.3 * val_recall
        elif self.model_type == "rf_detr":
            config = {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "lr_encoder": trial.suggest_float("lr_encoder", 1e-6, 1e-4, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                "clip_max_norm": trial.suggest_float("clip_max_norm", 0.05, 0.5),
                "epochs": 50, # extended training for better HPO
                "name": f"rfdetr_hpo_trial_{trial.number}"
            }
            val_map50 = self._train_and_evaluate_rfdetr(config)
            return val_map50
        else:
            raise NotImplementedError(f"HPO for {self.model_type} is not implemented.")

    def _train_and_evaluate_rfdetr(self, config):
        from src.models.rf_detr_trainer import RFDETRTrainer
        import csv
        
        output_dir = f"runs/detect/rfdetr_hpo/{config['name']}"
        base_config = {
            "epochs": config["epochs"],
            "batch_size": 8,
            "lr": config["lr"],
            "lr_encoder": config["lr_encoder"],
            "weight_decay": config["weight_decay"],
            "clip_max_norm": config["clip_max_norm"],
            "output_dir": output_dir,
            "wandb": False,
            "patience": 10,
            "num_workers": 0,
        }
        
        trainer = RFDETRTrainer(base_config)
        try:
            model = trainer.train(dataset_dir="dataset_final_resplit", model_size="small")
            
            # Extract metrics from model object first
            val_map50 = 0.0
            if hasattr(model, 'metrics') and hasattr(model.metrics, 'map50'):
                val_map50 = model.metrics.map50
            elif hasattr(model, 'eval_results') and 'map50' in model.eval_results:
                val_map50 = model.eval_results['map50']
            
            # Fallback: parse metrics.csv written by RF-DETR
            if val_map50 == 0.0:
                metrics_path = Path(output_dir) / "metrics.csv"
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]
                            for key in ['val_map50', 'map50', 'AP50', 'val/map50']:
                                if key in last_row:
                                    val_map50 = float(last_row[key])
                                    break
                
            del model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            return val_map50
        except Exception as e:
            print(f"RF-DETR Trial failed: {e}")
            return 0.0

    def _train_and_evaluate(self, config):
        from src.models.yolov8_trainer import YOLOTrainer
        
        # Merge with base config
        base_config = {
            "epochs": config["epochs"],
            "batch": 8,
            "workers": 0,
            "device": 0,
            "name": config["name"],
            "project": "runs/detect/gallstone_detection_hpo",
            "exist_ok": True,
            "lr0": config["lr0"],
            "lrf": config["lrf"],
            "weight_decay": config["weight_decay"],
            "box": config["box"],
            "cls": config["cls"],
            "dfl": config["dfl"],
            "optimizer": config["optimizer"],
            "patience": 10
        }
        
        trainer = YOLOTrainer(base_config)
        try:
            # We train from yolo26m to keep it fast
            model = trainer.train(data_yaml_path="data/processed/data.yaml", weights="yolo26m.pt")
            
            # Extract metrics from the model's trainer
            if hasattr(model, 'trainer') and model.trainer is not None:
                val_map50 = model.trainer.metrics.get('metrics/mAP50(B)', 0.0)
                val_recall = model.trainer.metrics.get('metrics/recall(B)', 0.0)
            else:
                # fallback if we can't get it directly
                val_map50, val_recall = 0.0, 0.0
                
            # Cleanup memory
            del model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            return val_map50, val_recall
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0, 0.0

    def run(self):
        study_name = f"gallstone_{self.model_type}_hpo"
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            study_name=study_name,
            storage=f"sqlite:///{self.model_type}_hpo_results.db",
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1, show_progress_bar=True)
        print(f"\nBest parameters found for {self.model_type}: {study.best_params}")
        return study.best_params

