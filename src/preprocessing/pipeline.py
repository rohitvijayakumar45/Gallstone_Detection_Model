"""Ultrasound preprocessing pipeline for training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class UltrasoundPreprocessor:
    """Speckle reduction, CLAHE, gamma normalization, and letterbox resize."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.target_size = int(config.get("resolution", 640))
        self.clahe_clip_limit = float(config.get("clahe_clip_limit", 2.0))
        self.clahe_tile_grid = tuple(config.get("clahe_tile_grid", (8, 8)))
        # Speckle denoiser: "bilateral" (legacy, fast) or "nlmeans" (better for US)
        self.denoiser = str(config.get("denoiser", "bilateral")).lower()
        self.nlm_h = float(config.get("nlm_h", 10.0))
        self.nlm_template = int(config.get("nlm_template_window", 7))
        self.nlm_search = int(config.get("nlm_search_window", 21))

    def process(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        if image is None:
            raise ValueError("image is None")

        processed, metadata = self.process_with_metadata(image)
        return processed, metadata["ratio"], metadata["pad"]

    def process_with_metadata(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Process image and return geometry needed to transform YOLO labels."""

        if image is None:
            raise ValueError("image is None")
        orig_h, orig_w = image.shape[:2]
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray, crop = self._crop_ultrasound_region(gray, return_crop=True)
        if self.denoiser == "nlmeans":
            denoised = cv2.fastNlMeansDenoising(
                gray,
                None,
                h=self.nlm_h,
                templateWindowSize=self.nlm_template,
                searchWindowSize=self.nlm_search,
            )
        else:
            denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        enhanced = clahe.apply(denoised)
        gamma_corrected = self._apply_gamma(enhanced, self._auto_gamma(enhanced))

        p1, p99 = np.percentile(gamma_corrected, [1, 99])
        if p99 <= p1:
            normalized = gamma_corrected.astype(np.uint8)
        else:
            clipped = np.clip(gamma_corrected, p1, p99)
            normalized = ((clipped - p1) / (p99 - p1) * 255).astype(np.uint8)

        bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        resized, ratio, pad = self._letterbox_resize(bgr, self.target_size)
        metadata = {
            "orig_shape": (orig_h, orig_w),
            "crop": crop,
            "ratio": ratio,
            "pad": pad,
            "target_size": self.target_size,
        }
        return resized, metadata

    def _crop_ultrasound_region(self, gray: np.ndarray, return_crop: bool = False):
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            crop = (0, 0, gray.shape[1], gray.shape[0])
            return (gray, crop) if return_crop else gray

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.05 * gray.shape[0] * gray.shape[1]:
            crop = (0, 0, gray.shape[1], gray.shape[0])
            return (gray, crop) if return_crop else gray

        x, y, w, h = cv2.boundingRect(largest)
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)
        cropped = gray[y : y + h, x : x + w]
        crop = (x, y, w, h)
        return (cropped, crop) if return_crop else cropped

    def _auto_gamma(self, image: np.ndarray) -> float:
        mean_val = float(np.mean(image) / 255.0)
        if mean_val < 0.3:
            return 0.7
        if mean_val > 0.7:
            return 1.3
        return 1.0

    def _apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / max(gamma, 1e-6)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
            np.uint8
        )
        return cv2.LUT(image, table)

    def _letterbox_resize(
        self, image: np.ndarray, target_size: int
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = image.shape[:2]
        ratio = min(target_size / h, target_size / w)
        new_h, new_w = max(1, int(h * ratio)), max(1, int(w * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
        return canvas, ratio, (pad_w, pad_h)


def preprocess_dataset(
    data_dir: str | Path,
    preprocessor: UltrasoundPreprocessor,
    output_dir: str | Path = "data/processed",
    max_images: int | None = None,
) -> dict[str, int]:
    """Preprocess YOLO-style split images and transform bbox/polygon labels."""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    counts: dict[str, int] = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for split in ("train", "val", "test"):
        src_img_dir = data_dir / split / "images"
        src_label_dir = data_dir / split / "labels"
        if not src_img_dir.exists():
            continue
        dst_split = "valid" if split == "val" else split
        dst_img_dir = output_dir / dst_split / "images"
        dst_label_dir = output_dir / dst_split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for image_path in src_img_dir.iterdir():
            if image_path.suffix.lower() not in image_exts:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            processed, metadata = preprocessor.process_with_metadata(image)
            cv2.imwrite(str(dst_img_dir / image_path.name), processed)
            label = src_label_dir / f"{image_path.stem}.txt"
            if label.exists():
                transformed = transform_yolo_label(label.read_text(encoding="utf-8"), metadata)
                (dst_label_dir / label.name).write_text(transformed, encoding="utf-8")
            n += 1
            if max_images and n >= max_images:
                break
        counts[split] = n
    return counts


def transform_yolo_label(label_text: str, metadata: dict[str, Any]) -> str:
    orig_h, orig_w = metadata["orig_shape"]
    crop_x, crop_y, _crop_w, _crop_h = metadata["crop"]
    ratio = metadata["ratio"]
    pad_w, pad_h = metadata["pad"]
    target = metadata["target_size"]
    rows: list[str] = []

    for raw in label_text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        cls = parts[0]
        nums = [float(x) for x in parts[1:]]
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            x1 = (cx - bw / 2) * orig_w
            y1 = (cy - bh / 2) * orig_h
            x2 = (cx + bw / 2) * orig_w
            y2 = (cy + bh / 2) * orig_h
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            pts = [(nums[i] * orig_w, nums[i + 1] * orig_h) for i in range(0, len(nums), 2)]
        else:
            continue

        out: list[float] = []
        for x, y in pts:
            tx = ((x - crop_x) * ratio + pad_w) / target
            ty = ((y - crop_y) * ratio + pad_h) / target
            out.extend([min(max(tx, 0.0), 1.0), min(max(ty, 0.0), 1.0)])

        xs = out[0::2]
        ys = out[1::2]
        if max(xs) - min(xs) <= 1e-5 or max(ys) - min(ys) <= 1e-5:
            continue
        rows.append(" ".join([cls, *[f"{v:.6f}" for v in out]]))
    return "\n".join(rows) + ("\n" if rows else "")
