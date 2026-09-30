from __future__ import annotations


class YOLOTrainer:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def train(self, data_yaml_path: str, weights: str = "yolo26m.pt"):
        try:
            from ultralytics.utils import SETTINGS

            SETTINGS["mlflow"] = False
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Install `ultralytics>=8.3.0` to train YOLO.") from exc

        import os
        from pathlib import Path

        os.environ["ULTRALYTICS_MLFLOW"] = "False"
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        model = YOLO(weights)
        model.train(
            data=data_yaml_path,
            epochs=self.config.get("epochs", 200),
            patience=self.config.get("patience", 50),
            imgsz=self.config.get("imgsz", 640),
            batch=self.config.get("batch", 8),
            workers=self.config.get("workers", 4),
            device=self.config.get("device", 0),
            optimizer="AdamW",
            lr0=self.config.get("lr0", 0.001),
            lrf=self.config.get("lrf", 0.01),
            momentum=0.937,
            weight_decay=self.config.get("weight_decay", 0.0005),
            warmup_epochs=5,
            box=self.config.get("box", 7.5),
            cls=self.config.get("cls", 0.5),
            dfl=self.config.get("dfl", 1.5),
            augment=True,
            conf=self.config.get("conf", 0.25),
            iou=self.config.get("iou", 0.45),
            close_mosaic=20,
            amp=True,
            project=str(Path(self.config.get("project", "runs/detect/gallstone_detection")).resolve()),
            name=self.config.get("name", weights.replace(".pt", "_baseline")),
            save=True,
            plots=True,
            exist_ok=self.config.get("exist_ok", True),
        )
        return model

    def val(self, data_yaml_path: str, weights: str, end2end: bool | None = None):
        try:
            from ultralytics.utils import SETTINGS

            SETTINGS["mlflow"] = False
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Install latest `ultralytics` to validate YOLO.") from exc

        import os

        os.environ["ULTRALYTICS_MLFLOW"] = "False"
        os.environ.pop("MLFLOW_TRACKING_URI", None)
        model = YOLO(weights)
        kwargs = {"data": data_yaml_path, "imgsz": self.config.get("imgsz", 640), "device": self.config.get("device", 0)}
        if end2end is not None:
            kwargs["end2end"] = end2end
        return model.val(**kwargs)


YOLOv8Trainer = YOLOTrainer
