from .rf_detr_trainer import RFDETRTrainer
from .yolov8_trainer import YOLOv8Trainer


def build_model(model_type: str, config: dict | None = None):
    if model_type == "rf_detr":
        return RFDETRTrainer(config).build_model(config.get("model_size", "base") if config else "base")
    if model_type in {"yolov8", "yolo11"}:
        weights = "yolo11m.pt" if model_type == "yolo11" else "yolov8m.pt"
        from ultralytics import YOLO

        return YOLO(weights)
    raise ValueError(f"Unknown model_type: {model_type}")
