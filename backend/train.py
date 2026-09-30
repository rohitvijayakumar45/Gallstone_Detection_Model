"""
train.py
--------
Fine-tune YOLO26 on the gallstone/gallbladder ultrasound dataset.

Dataset has DETECTION labels (bounding boxes), so we train YOLO26 detection.
If you run convert_boxes_to_masks.py first, switch --model to yolo26s-seg.

Classes:
  0 = Bladder      1 = Gallbladder (primary target)      2 = Vessels

Usage
-----
  python train.py                        # detection, CPU, 100 epochs
  python train.py --device 0             # GPU
  python train.py --model yolo26s-seg    # segmentation (run SAM converter first)
  python train.py --epochs 150 --batch 16

Run setup first:
  python setup_dataset.py
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO26-seg for gallstone detection")
    p.add_argument("--model",   default="yolo26s",  help="Base model variant (use yolo26s-seg only if you ran convert_boxes_to_masks.py)")
    p.add_argument("--data",    default="dataset/data.yaml")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=8,
                   help="Reduce to 4 if CUDA out-of-memory")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--device",  default="0",
                   help="'cpu', '0' (first GPU), 'cuda', or 'mps'")
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name",    default="gallstone_yolo26")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: dataset not found at '{data_path}'.")
        print("Download from Roboflow Universe:")
        print("  1. Go to universe.roboflow.com")
        print("  2. Search 'gallstone ultrasound'")
        print("  3. Export → Format: YOLOv8, Augmentation: OFF, Split: 80/15/5")
        print("  4. Unzip into backend/dataset/")
        return

    # Load YOLO26-seg (downloads pretrained weights automatically on first run)
    model_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    print(f"\nStarting training on device={args.device}")
    print(f"  Model   : {model_name}")
    print(f"  Dataset : {data_path}")
    print(f"  Epochs  : {args.epochs}  |  Batch: {args.batch}  |  Imgsz: {args.imgsz}\n")

    results = model.train(
        data    = str(data_path),
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = args.device,
        project = args.project,
        name    = args.name,

        # Early stopping — stops if no improvement for 25 epochs
        patience = 25,

        # Learning rate schedule
        lr0  = 0.01,
        lrf  = 0.01,

        # ── Online augmentation (applied fresh every epoch) ──────────────
        # These replace the need for pre-augmented images from Roboflow.
        mosaic      = 1.0,    # 4-image mosaic (very effective for small datasets)
        mixup       = 0.15,   # MixUp blending
        copy_paste  = 0.15,   # copy-paste segmentation augmentation
        degrees     = 10.0,   # random rotation ±10° (safe for ultrasound)
        translate   = 0.1,
        scale       = 0.5,
        flipud      = 0.3,
        fliplr      = 0.5,
        hsv_h       = 0.015,  # hue jitter (minimal — ultrasound is greyscale)
        hsv_s       = 0.5,
        hsv_v       = 0.4,

        # Save best weights
        save    = True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    dest         = Path("weights/gallstone_seg.pt")

    if best_weights.exists():
        import shutil
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, dest)
        print(f"\nBest weights copied to {dest}")
        print("Run the backend: uvicorn main:app --reload --port 8000")
    else:
        print(f"\nTraining complete. Find weights at: {args.project}/{args.name}/weights/")

    # Print validation metrics
    print("\nValidation metrics:")
    print(f"  mAP50-95 (boxes) : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    print(f"  mAP50-95 (masks) : {results.results_dict.get('metrics/mAP50-95(M)', 'N/A'):.3f}")


if __name__ == "__main__":
    main()
