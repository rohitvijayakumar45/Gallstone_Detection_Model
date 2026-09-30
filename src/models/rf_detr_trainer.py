from __future__ import annotations

from pathlib import Path


class RFDETRTrainer:
    """RF-DETR wrapper with graceful fallback when dependency is unavailable."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model = None

    def build_model(self, model_size: str = "small"):
        try:
            from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall
        except Exception as exc:
            raise RuntimeError("Install `rfdetr>=0.3.0` to train RF-DETR.") from exc

        model_map = {"small": RFDETRSmall, "base": RFDETRBase, "large": RFDETRLarge}
        self.model = model_map[model_size]()
        return self.model

    def train(self, dataset_dir: str | Path, model_size: str = "base"):
        model = self.build_model(model_size)
        train_args = {
            "dataset_dir": str(dataset_dir),
            "epochs": self.config.get("epochs", 150),
            "batch_size": self.config.get("batch_size", 16),
            "lr": self.config.get("lr", 1e-4),
            "lr_encoder": self.config.get("lr_backbone", self.config.get("lr_encoder", 1e-5)),
            "weight_decay": self.config.get("weight_decay", 1e-4),
            "grad_accum_steps": self.config.get("grad_accumulation_steps", self.config.get("grad_accum_steps", 2)),
            "clip_max_norm": self.config.get("clip_max_norm", 0.1),
            "early_stopping": True,
            "early_stopping_patience": self.config.get("patience", 20),
            "output_dir": self.config.get("output_dir", "./weights/rf_detr/"),
            "wandb": self.config.get("use_wandb", self.config.get("wandb", True)),
            "mlflow": self.config.get("mlflow", False),
            "project": "gallstone_detection",
            "run": f"rfdetr_{model_size}_{self.config.get('resolution', 640)}",
            "dataset_file": self.config.get("dataset_file", "yolo"),
            "class_names": self.config.get("class_names", ["gallstone"]),
            "num_workers": self.config.get("num_workers", 2),
            "progress_bar": self.config.get("progress_bar", "tqdm"),
            "resume": self.config.get("resume"),
            "lr_scheduler": self.config.get("lr_scheduler", "cosine"),
            "eval_interval": self.config.get("eval_interval", 1),
            "checkpoint_interval": self.config.get("checkpoint_interval", 10),
        }
        train_args = {key: value for key, value in train_args.items() if value is not None}
        model.train(**train_args)
        return model
