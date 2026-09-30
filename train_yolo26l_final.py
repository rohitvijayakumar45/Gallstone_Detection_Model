"""
YOLOv26 Large — Full Training with HPO-Winning Parameters
Best Trial #16: mAP50=0.95978 (50 epochs)
Target: ≥97% mAP50 with 300 epochs, no early stopping.
"""
import multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()

import sys
import os

# ── Windows compatibility ────────────────────────────────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
cv2.setNumThreads(0)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ULTRALYTICS_MLFLOW"] = "False"
os.environ.pop("MLFLOW_TRACKING_URI", None)

from pathlib import Path


def main():
    from ultralytics.utils import SETTINGS
    SETTINGS["mlflow"] = False
    from ultralytics import YOLO

    # ── HPO-Winning Parameters (Trial #16, mAP50=0.95978 @ 50 epochs) ────
    WEIGHTS = "yolo26l.pt"
    DATA_YAML = "dataset_final_resplit/data.yaml"
    PROJECT = str(Path("runs/detect/gallstone_detection").resolve())
    NAME = "yolo26l_final"

    print("=" * 70)
    print("  YOLOv26 LARGE — Full Training (HPO-Winning Params)")
    print(f"  Weights:  {WEIGHTS}")
    print(f"  Dataset:  {DATA_YAML}")
    print(f"  Epochs:   300 (no early stopping)")
    print(f"  Output:   {PROJECT}/{NAME}")
    print("=" * 70)

    model = YOLO(WEIGHTS)
    model.train(
        data=DATA_YAML,
        epochs=300,
        patience=0,            # No early stopping — best weights auto-saved
        imgsz=640,
        batch=6,
        workers=4,
        device=0,
        optimizer="AdamW",

        # ── HPO-winning hyperparameters (Trial #16) ──────────────────────
        lr0=0.0003729790675877055,
        lrf=0.1848577362852768,
        momentum=0.937,
        weight_decay=0.002179440885696096,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=6.31751915780746,
        cls=0.600366886995146,
        dfl=2.228632819733598,
        cos_lr=False,
        close_mosaic=15,
        amp=True,
        nbs=64,

        # ── Augmentation (HPO-tuned) ─────────────────────────────────────
        augment=True,
        mixup=0.02553818035946693,
        copy_paste=0.09425586863956208,
        dropout=0.12209905583667471,
        mosaic=1.0,
        fliplr=0.5,
        scale=0.5,
        erasing=0.4,
        auto_augment="randaugment",

        # ── Output ───────────────────────────────────────────────────────
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        seed=42,
        deterministic=True,
    )

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print(f"  Best weights: {PROJECT}/{NAME}/weights/best.pt")
    print(f"  Last weights: {PROJECT}/{NAME}/weights/last.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
