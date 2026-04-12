"""
merge_datasets.py
-----------------
Merges two Roboflow datasets into one unified training dataset.

Dataset A  (826 images) — Anatomical structures (detection format, bounding boxes)
  Classes: Bladder · Gallbladder · Vessels
  Source : C:/Users/RohitLocal/Downloads/gallstone -v1-using YOLOV 11.yolo26

Dataset B  (462 images) — Gallstone segmentation (polygon format)  ← primary
  Classes: gallstone
  Source : C:/Users/RohitLocal/Downloads/Gallstone-Detection.yolo26

Merge strategy
--------------
Unified class map:
  Class 0 → gallstone      (from Dataset B — polygon labels, kept as-is)
  Class 1 → Gallbladder    (from Dataset A class 1 only — box converted to 4-pt polygon)

  Dataset A classes 0 (Bladder) and 2 (Vessels) are excluded — not relevant
  for gallstone detection and would add noise.

Label format normalisation:
  Dataset B labels → already polygon, kept unchanged.
  Dataset A Gallbladder labels → box (cx cy w h) converted to 4-corner polygon.

Result: ~1288 images, 2 classes, all segmentation-format labels
  Split: 70 / 20 / 10  train / valid / test

Run:
    python merge_datasets.py
"""

import os
import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DS_A = Path(r"C:/Users/RohitLocal/Downloads/gallstone -v1-using YOLOV 11.yolo26")
DS_B = Path(r"C:/Users/RohitLocal/Downloads/Gallstone-Detection.yolo26")

DEST        = Path("dataset")
RANDOM_SEED = 42
SPLITS      = {"train": 0.70, "valid": 0.20, "test": 0.10}

# Unified class system
CLASS_NAMES = ["gallstone", "Gallbladder"]
# DS_B class 0 (gallstone)    → unified class 0
# DS_A class 1 (Gallbladder)  → unified class 1
# DS_A class 0 (Bladder)      → SKIP
# DS_A class 2 (Vessels)      → SKIP


# ── Helpers ───────────────────────────────────────────────────────────────────

def box_to_polygon_label(line: str) -> str | None:
    """
    Convert a YOLO detection label line (cls cx cy w h) to a
    4-corner polygon label (cls x0y0 x1y0 x1y1 x0y1).
    Returns None if the line is invalid.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls, cx, cy, w, h = parts
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    # 4 corners: TL TR BR BL (normalised)
    return f"{cls} {x0:.6f} {y0:.6f} {x1:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x0:.6f} {y1:.6f}"


def is_segmentation_line(line: str) -> bool:
    return len(line.strip().split()) > 5


def collect_images(folder: Path) -> list[Path]:
    imgs = []
    img_dir = folder / "train" / "images"
    if img_dir.exists():
        imgs += list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    # Also check valid/test in case they exist
    for split in ("valid", "test"):
        d = folder / split / "images"
        if d.exists():
            imgs += list(d.glob("*.jpg")) + list(d.glob("*.png"))
    return sorted(set(imgs))


def find_label(img_path: Path, dataset_root: Path) -> Path | None:
    """Find the label file for an image across all split folders."""
    for split in ("train", "valid", "test"):
        lbl = dataset_root / split / "labels" / (img_path.stem + ".txt")
        if lbl.exists():
            return lbl
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    for ds, name in [(DS_A, "Dataset A"), (DS_B, "Dataset B")]:
        if not ds.exists():
            print(f"ERROR: {name} not found at:\n  {ds}")
            return

    print("=== Merging Datasets ===\n")

    # Clear and recreate destination
    if DEST.exists():
        shutil.rmtree(DEST)
    for split in SPLITS:
        (DEST / split / "images").mkdir(parents=True)
        (DEST / split / "labels").mkdir(parents=True)

    # ── Collect all entries ───────────────────────────────────────────────────
    entries = []   # list of (img_path, label_lines_transformed, source_tag)

    # ── Dataset B — gallstone polygon labels (primary) ───────────────────────
    b_images = collect_images(DS_B)
    b_ok, b_skip = 0, 0
    for img in b_images:
        lbl = find_label(img, DS_B)
        if lbl is None:
            b_skip += 1
            continue
        # Empty label file = background image — include with empty lines
        if lbl.stat().st_size == 0:
            entries.append((img, [], "DS_B_bg"))
            b_ok += 1
            continue
        raw = lbl.read_text().strip().splitlines()
        out_lines = []
        for line in raw:
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls == 0:   # gallstone → unified class 0
                new_line = "0 " + " ".join(parts[1:])
                out_lines.append(new_line)
        if out_lines:
            entries.append((img, out_lines, "DS_B"))
            b_ok += 1
        else:
            # Empty label = background image (no gallstone visible).
            # Include it — teaches the model to output nothing on clean scans.
            entries.append((img, [], "DS_B_bg"))
            b_ok += 1

    print(f"Dataset B (gallstone polygons): {b_ok} usable ({b_skip} had no label file)")

    # ── Dataset A — Gallbladder class only, box → polygon ────────────────────
    a_images = collect_images(DS_A)
    a_ok, a_skip = 0, 0
    for img in a_images:
        lbl = find_label(img, DS_A)
        if lbl is None:
            a_skip += 1
            continue
        raw = lbl.read_text().strip().splitlines()
        out_lines = []
        for line in raw:
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls == 1:   # Gallbladder → unified class 1
                if is_segmentation_line(line):
                    # Already polygon (shouldn't happen but handle it)
                    out_lines.append("1 " + " ".join(parts[1:]))
                else:
                    # Convert box to 4-corner polygon
                    converted = box_to_polygon_label(line)
                    if converted:
                        # Remap class 1 → unified class 1 (already 1, no change)
                        out_lines.append(converted)
            # cls 0 (Bladder) and cls 2 (Vessels) → skipped
        if out_lines:
            entries.append((img, out_lines, "DS_A"))
            a_ok += 1
        else:
            a_skip += 1

    print(f"Dataset A (Gallbladder boxes):  {a_ok} usable, {a_skip} skipped (Bladder/Vessels excluded)")

    total = len(entries)
    print(f"\nTotal usable images: {total}")

    # ── Shuffle and split ─────────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    random.shuffle(entries)

    n_train = int(total * SPLITS["train"])
    n_valid = int(total * SPLITS["valid"])

    split_entries = {
        "train": entries[:n_train],
        "valid": entries[n_train:n_train + n_valid],
        "test":  entries[n_train + n_valid:],
    }

    print("\nSplit:")
    for name, ents in split_entries.items():
        ds_b_count  = sum(1 for _, _, tag in ents if tag == "DS_B")
        ds_b_bg     = sum(1 for _, _, tag in ents if tag == "DS_B_bg")
        ds_a_count  = sum(1 for _, _, tag in ents if tag == "DS_A")
        print(f"  {name:6}: {len(ents):4} images  (gallstone: {ds_b_count}, background: {ds_b_bg}, gallbladder: {ds_a_count})")

    # ── Copy and write ────────────────────────────────────────────────────────
    print("\nCopying files…")
    for split_name, ents in split_entries.items():
        img_dir = DEST / split_name / "images"
        lbl_dir = DEST / split_name / "labels"

        for img_path, out_lines, _ in ents:
            # Avoid filename collisions by prefixing with source hash
            stem = img_path.stem
            dest_img = img_dir / img_path.name
            dest_lbl = lbl_dir / (stem + ".txt")

            # Handle duplicate filenames (both datasets may share stems)
            if dest_img.exists():
                stem = stem + "_b"
                dest_img = img_dir / (stem + img_path.suffix)
                dest_lbl = lbl_dir / (stem + ".txt")

            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(out_lines))

    # ── Write data.yaml ───────────────────────────────────────────────────────
    abs_dest = DEST.resolve().as_posix()
    yaml = f"""# GallScan AI v3 — Merged Dataset
# Auto-generated by merge_datasets.py
#
# Class 0 (gallstone)   — polygon segmentation masks from Dataset B (462 images)
# Class 1 (Gallbladder) — 4-point polygon from Dataset A bounding boxes (826 images)
#
# Total: {total} images  |  Split: 70/20/10

path:  {abs_dest}
train: train/images
val:   valid/images
test:  test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    yaml_path = DEST / "data.yaml"
    yaml_path.write_text(yaml)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Merged dataset ready.\n")
    for split in ("train", "valid", "test"):
        n_img = len(list((DEST / split / "images").glob("*.jpg"))) + \
                len(list((DEST / split / "images").glob("*.png")))
        n_lbl = len(list((DEST / split / "labels").glob("*.txt")))
        print(f"  {split:6}: {n_img} images, {n_lbl} labels")

    print(f"\nClasses : {CLASS_NAMES}")
    print(f"YAML    : {yaml_path.resolve()}")
    print(f"""
Training commands:
  Segmentation (recommended — polygon labels available):
    python train.py --model yolo26s-seg --data dataset/data.yaml

  Detection only (faster):
    python train.py --model yolo26s --data dataset/data.yaml
""")


if __name__ == "__main__":
    main()
