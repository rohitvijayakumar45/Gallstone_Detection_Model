"""
convert_boxes_to_masks.py
--------------------------
Convert a YOLO detection dataset (bounding-box labels) into a YOLO
segmentation dataset (polygon labels) using Meta's Segment Anything Model (SAM).

Run this ONLY if your Roboflow export has detection-only labels.
If you downloaded a segmentation dataset, skip this script entirely.

Usage
-----
  pip install segment-anything
  # Download SAM checkpoint from Meta:
  #   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  python convert_boxes_to_masks.py --split train
  python convert_boxes_to_masks.py --split valid

After running, update dataset/data.yaml to point label dirs at the new _seg/ dirs.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="train", choices=["train", "valid", "test"])
    p.add_argument("--dataset",    default="dataset")
    p.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    p.add_argument("--model-type", default="vit_b", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


def box_to_polygon(mask: np.ndarray, W: int, H: int) -> list[float] | None:
    """Convert a binary mask to a normalised YOLO polygon string."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    # Use the largest contour
    pts = max(contours, key=cv2.contourArea).squeeze()
    if pts.ndim < 2 or len(pts) < 3:
        return None
    # Normalise to [0, 1]
    return [coord for pt in pts for coord in (pt[0] / W, pt[1] / H)]


def main():
    args = parse_args()

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("Install SAM: pip install segment-anything")
        return

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"SAM checkpoint not found at '{checkpoint}'.")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return

    print(f"Loading SAM ({args.model_type}) from {checkpoint} on {args.device}…")
    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint))
    sam.to(args.device)
    predictor = SamPredictor(sam)

    dataset   = Path(args.dataset)
    img_dir   = dataset / args.split / "images"
    lbl_dir   = dataset / args.split / "labels"
    out_dir   = dataset / args.split / "labels_seg"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    print(f"Processing {len(images)} images in '{img_dir}'…")

    skipped = 0
    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        out_path = out_dir / lbl_path.name

        if not lbl_path.exists():
            skipped += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Could not read {img_path.name}, skipping.")
            skipped += 1
            continue

        H, W = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        lines_in  = lbl_path.read_text().strip().splitlines()
        lines_out = []

        for line in lines_in:
            parts = line.split()
            if len(parts) < 5:
                continue

            cls, cx, cy, bw, bh = (float(x) for x in parts[:5])

            # Convert normalised box to pixel coords
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)
            box = np.array([[x1, y1, x2, y2]], dtype=float)

            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            poly = box_to_polygon(masks[0], W, H)

            if poly is None:
                print(f"  No contour for box in {img_path.name}, keeping box label.")
                lines_out.append(line)
                continue

            seg_coords = " ".join(f"{v:.6f}" for v in poly)
            lines_out.append(f"{int(cls)} {seg_coords}")

        out_path.write_text("\n".join(lines_out))

    print(f"\nDone. Output labels at: {out_dir}")
    print(f"Skipped: {skipped} images (no label file or unreadable)")
    print("\nNext step: update dataset/data.yaml labels path to point at labels_seg/")


if __name__ == "__main__":
    main()
