from pathlib import Path


def resolve_data_dir(data_dir):
    path = Path(data_dir)
    if (path / "train" / "images").exists():
        return path
    fallback = Path("dataset_final")
    if fallback.exists():
        return fallback
    return path


def dataset_counts(data_dir):
    data_dir = Path(data_dir)
    split_dirs = {"train": "train", "val": "valid" if (data_dir / "valid").exists() else "val", "test": "test"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return {
        split: sum(1 for p in (data_dir / folder / "images").glob("*") if p.suffix.lower() in image_exts)
        if (data_dir / folder / "images").exists()
        else 0
        for split, folder in split_dirs.items()
    }
