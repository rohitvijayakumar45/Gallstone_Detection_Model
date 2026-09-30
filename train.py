from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from src.preprocessing import UltrasoundPreprocessor, preprocess_dataset
from src.utils.data_loader import dataset_counts, resolve_data_dir

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def write_project_data_yaml(source_dir: Path, target_dir: Path = Path("data")) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    src_yaml = source_dir / "data.yaml"
    dst_yaml = target_dir / "data.yaml"
    if src_yaml.exists():
        shutil.copyfile(src_yaml, dst_yaml)
    else:
        dst_yaml.write_text(
            "path: ../dataset_final\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: gallstone\n",
            encoding="utf-8",
        )
    return dst_yaml


def write_rfdetr_data_yaml(dataset_dir: Path) -> Path:
    dataset_root = dataset_dir.resolve().as_posix()
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {dataset_root}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n  0: gallstone\n",
        encoding="utf-8",
    )
    return yaml_path


def validate_yolo_dataset(data_dir: Path) -> dict:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    counts = {
        split: sum(
            1
            for p in (data_dir / split / "images").glob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )
        if (data_dir / split / "images").exists()
        else 0
        for split in ("train", "val", "test")
    }
    label_stats = {}
    for split in ("train", "val", "test"):
        label_dir = data_dir / split / "labels"
        files = list(label_dir.glob("*.txt")) if label_dir.exists() else []
        polygon_rows = 0
        bbox_rows = 0
        bad_rows = 0
        for label in files[:5000]:
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                if len(parts) == 5:
                    bbox_rows += 1
                elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                    polygon_rows += 1
                else:
                    bad_rows += 1
        label_stats[split] = {
            "label_files": len(files),
            "bbox_rows": bbox_rows,
            "polygon_rows": polygon_rows,
            "bad_rows": bad_rows,
        }
    return {"image_counts": counts, "label_stats": label_stats}


def phase1(
    data_dir: Path,
    preprocess_limit: int | None,
    train_rfdetr: bool,
    epochs: int,
    model_size: str,
    batch_size: int,
    grad_accum_steps: int,
    num_workers: int,
    resume: str | None,
    output_dir: str,
    lr: float,
    lr_encoder: float,
    skip_preprocess: bool,
):
    print("PHASE 1: data validation + preprocessing + RF-DETR availability check")
    data_yaml = write_project_data_yaml(data_dir)
    validation = validate_yolo_dataset(data_dir)
    Path("mlruns").mkdir(exist_ok=True)
    Path("weights/rf_detr").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("mlruns/phase1_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))

    preprocessor = UltrasoundPreprocessor({"resolution": 640, "clahe_clip_limit": 2.0})
    max_images = 12 if preprocess_limit == 0 else preprocess_limit
    if skip_preprocess:
        counts = dataset_counts(Path("data/processed"))
    else:
        counts = preprocess_dataset(data_dir, preprocessor, output_dir="data/processed", max_images=max_images)
    rfdetr_yaml = write_rfdetr_data_yaml(Path("data/processed"))
    mode = "existing" if skip_preprocess else ("full" if max_images is None else f"limit={max_images}")
    print(f"Preprocessed set ({mode}): {counts}")
    print(f"RF-DETR YOLO yaml: {rfdetr_yaml}")

    try:
        from src.models.rf_detr_trainer import RFDETRTrainer

        trainer = RFDETRTrainer({"epochs": epochs, "use_wandb": False})
        trainer.build_model("small")
        print("RF-DETR import/build: ok")
        if train_rfdetr:
            print(f"Starting RF-DETR {model_size} training for {epochs} epoch(s)")
            trainer = RFDETRTrainer(
                {
                    "epochs": epochs,
                    "use_wandb": False,
                    "batch_size": batch_size,
                    "grad_accum_steps": grad_accum_steps,
                    "num_workers": num_workers,
                    "output_dir": output_dir,
                    "dataset_file": "yolo",
                    "resume": resume,
                    "lr": lr,
                    "lr_encoder": lr_encoder,
                    "lr_scheduler": "cosine",
                    "eval_interval": 1,
                }
            )
            trainer.train("data/processed", model_size=model_size)
    except Exception as exc:
        print(f"RF-DETR training skipped: {exc}")
        print("Install requirements + GPU, then run full training with RFDETRTrainer.train().")

    print(f"Phase 1 ready. Data yaml: {data_yaml}")


def phase2(model_type="yolo"):
    print(f"PHASE 2: Running {model_type.upper()} Hyperparameter Optimization.")
    from src.training.hyperparameter_tuner import HyperparameterTuner
    tuner = HyperparameterTuner(model_type=model_type, n_trials=30)
    best_params = tuner.run()
    print(f"Best {model_type.upper()} parameters found: {best_params}")
def phase3(
    data_dir: Path,
    train_yolo: bool,
    yolo_weights: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    val_yolo: str | None,
    end2end: bool | None,
    yolo_lr0: float | None,
    yolo_weight_decay: float | None,
    yolo_lrf: float | None,
    yolo_box: float | None,
    yolo_cls: float | None,
    yolo_dfl: float | None,
):
    print("PHASE 3: ensemble scaffold ready.")
    yaml_path = data_dir / "data.yaml"
    if data_dir == Path("data/processed"):
        yaml_path = write_rfdetr_data_yaml(data_dir)
    print(f"YOLO data yaml: {yaml_path}")
    from src.models.yolov8_trainer import YOLOTrainer

    trainer_config = {
        "epochs": epochs,
        "batch": batch_size,
        "workers": num_workers,
        "device": 0,
        "name": yolo_weights.replace(".pt", "_gallstone"),
        "project": "runs/detect/gallstone_detection",
        "exist_ok": True,
    }

    # Add custom YOLO parameters if provided
    if yolo_lr0 is not None:
        trainer_config["lr0"] = yolo_lr0
    if yolo_weight_decay is not None:
        trainer_config["weight_decay"] = yolo_weight_decay
    if yolo_lrf is not None:
        trainer_config["lrf"] = yolo_lrf
    if yolo_box is not None:
        trainer_config["box"] = yolo_box
    if yolo_cls is not None:
        trainer_config["cls"] = yolo_cls
    if yolo_dfl is not None:
        trainer_config["dfl"] = yolo_dfl

    trainer = YOLOTrainer(trainer_config)
    if train_yolo:
        trainer.train(str(yaml_path), weights=yolo_weights)
    if val_yolo:
        trainer = YOLOTrainer(
            {
                "device": 0,
            }
        )
        trainer.val(str(yaml_path), weights=val_yolo, end2end=end2end)


def phase4():
    print("PHASE 4: TTA/explainability/report scaffold ready.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--preprocess-limit", type=int, default=0, help="0 = smoke test 12 images; set -1 for all")
    parser.add_argument("--train-rfdetr", action="store_true", help="Launch real RF-DETR training after preprocessing")
    parser.add_argument("--epochs", type=int, default=1, help="RF-DETR epochs when --train-rfdetr is set")
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="small")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./weights/rf_detr/")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--train-yolo", action="store_true")
    parser.add_argument("--val-yolo", type=str, default=None)
    parser.add_argument("--end2end", choices=["true", "false", "auto"], default="auto")
    parser.add_argument("--yolo-lr0", type=float, default=None, help="YOLO initial learning rate")
    parser.add_argument("--yolo-lrf", type=float, default=None, help="YOLO final LR fraction")
    parser.add_argument("--yolo-weight-decay", type=float, default=None, help="YOLO weight decay")
    parser.add_argument("--yolo-box", type=float, default=None, help="YOLO box loss gain")
    parser.add_argument("--yolo-cls", type=float, default=None, help="YOLO cls loss gain")
    parser.add_argument("--yolo-dfl", type=float, default=None, help="YOLO dfl loss gain")
    parser.add_argument(
        "--yolo-weights",
        choices=[
            "yolo26n.pt",
            "yolo26s.pt",
            "yolo26m.pt",
            "yolo26l.pt",
            "yolo26x.pt",
            "yolov8m.pt",
            "yolo11m.pt",
            "yolov8s.pt",
            "yolo11s.pt",
        ],
        default="yolo26m.pt",
    )
    parser.add_argument("--run-through", action="store_true", help="Run all phases from 1 through --phase")
    parser.add_argument("--hpo-model", choices=["yolo", "rf_detr"], default="yolo", help="Model to run HPO for in Phase 2")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    preprocess_limit = None if args.preprocess_limit < 0 else args.preprocess_limit
    if args.run_through and args.phase >= 1 or args.phase == 1:
        phase1(
            data_dir,
            preprocess_limit,
            args.train_rfdetr,
            args.epochs,
            args.model_size,
            args.batch_size,
            args.grad_accum_steps,
            args.num_workers,
            args.resume,
            args.output_dir,
            args.lr,
            args.lr_encoder,
            args.skip_preprocess,
        )
    if args.run_through and args.phase >= 2 or args.phase == 2:
        phase2(model_type=args.hpo_model)
    if args.run_through and args.phase >= 3 or args.phase == 3:
        end2end = None if args.end2end == "auto" else args.end2end == "true"
        phase3(
            data_dir,
            args.train_yolo,
            args.yolo_weights,
            args.epochs,
            args.batch_size,
            args.num_workers,
            args.val_yolo,
            end2end,
            args.yolo_lr0,
            args.yolo_weight_decay,
            args.yolo_lrf,
            args.yolo_box,
            args.yolo_cls,
            args.yolo_dfl,
        )
    if args.run_through and args.phase >= 4 or args.phase == 4:
        phase4()


if __name__ == "__main__":
    main()
