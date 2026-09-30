"""Fit per-detector temperature scaling and post-fusion isotonic regression.

Pipeline:
  1. Load val split (default: dataset_final_resplit/val, 368 images).
  2. Run each detector at a permissive threshold (0.01) to collect all
     candidate confidences + IoU-matched TP/FP labels vs polygon-derived GT.
  3. Fit per-model TemperatureScaler on (raw_conf, is_TP).
  4. Run ensemble (WBF + shadow) at 0.01, collect fused confidences +
     is_TP, fit IsotonicFusion.
  5. Persist to production_models/ + emit calibration report JSON with
     before/after ECE/MCE/Brier/NLL.

Usage:
  python scripts/fit_calibration.py \
      --data-dir dataset_final_resplit --split val \
      --yolo-weights production_models/yolo_best.pt \
      --rfdetr-weights production_models/rfdetr_best.pth \
      --out-dir production_models --report runs/final_eval/calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.calibration import (
    IsotonicFusion,
    TemperatureScaler,
    adaptive_calibration_error,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_bins,
)
from src.evaluation.label_io import iter_split
from src.evaluation.shadow_analyzer import ShadowAnalyzer
from src.models.ensemble import GallstoneEnsemble


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (aa + ab - inter + 1e-9))


def _match_and_label(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                     gt_boxes: np.ndarray, iou_thresh: float = 0.5):
    """Greedy IoU matching per image. Returns (labels_1_for_TP, matched_gt_indices)."""
    labels = np.zeros(len(pred_boxes), dtype=np.int64)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return labels
    order = np.argsort(-pred_scores)
    matched: set[int] = set()
    for idx in order:
        best_iou, best_j = 0.0, -1
        for j in range(len(gt_boxes)):
            if j in matched:
                continue
            iou = _iou(pred_boxes[idx], gt_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            labels[idx] = 1
            matched.add(best_j)
    return labels


def _load_yolo(path: str):
    from ultralytics import YOLO
    return YOLO(path)


def _load_rfdetr(path: str):
    from rfdetr import RFDETRLarge
    m = RFDETRLarge(pretrain_weights=path)
    try:
        m.optimize_for_inference()
    except Exception:
        pass
    return m


def _yolo_predict(model, img_bgr: np.ndarray, conf: float):
    res = model.predict(source=img_bgr, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
    return xyxy, scores


def _rfdetr_predict(model, img_bgr: np.ndarray, conf: float):
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(rgb, threshold=conf)
    if hasattr(res, "xyxy"):
        boxes = np.asarray(res.xyxy, dtype=np.float32)
        scores = np.asarray(res.confidence, dtype=np.float32)
    elif isinstance(res, tuple) and len(res) == 3:
        boxes, scores, _ = res
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
    else:
        boxes, scores = np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    if boxes.size and boxes.max() <= 1.05:
        h, w = img_bgr.shape[:2]
        boxes = boxes * np.array([w, h, w, h], dtype=np.float32)
    return boxes, scores


def collect_pairs(model_predict, split_dir: Path, conf_floor: float = 0.01):
    all_scores, all_labels = [], []
    for _stem, img, gt in tqdm(list(iter_split(split_dir)), desc=split_dir.name):
        boxes, scores = model_predict(img, conf_floor)
        if len(boxes) == 0:
            continue
        labels = _match_and_label(boxes, scores, gt)
        all_scores.append(scores)
        all_labels.append(labels)
    if not all_scores:
        return np.empty(0), np.empty(0)
    return np.concatenate(all_scores), np.concatenate(all_labels)


def _score_block(name: str, probs: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "name": name,
        "n": int(len(probs)),
        "ece": expected_calibration_error(probs, labels, n_bins=15),
        "adaptive_ece": adaptive_calibration_error(probs, labels, n_bins=15),
        "mce": maximum_calibration_error(probs, labels, n_bins=15),
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
        "reliability": reliability_bins(probs, labels, n_bins=10),
    }


def main():
    p = argparse.ArgumentParser(description="Fit calibration artefacts")
    p.add_argument("--data-dir", default="dataset_final_resplit")
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--yolo-weights", default="production_models/yolo_best.pt")
    p.add_argument("--rfdetr-weights", default="production_models/rfdetr_best.pth")
    p.add_argument("--out-dir", default="production_models")
    p.add_argument("--report", default="runs/final_eval/calibration.json")
    p.add_argument("--conf-floor", type=float, default=0.01)
    args = p.parse_args()

    split_dir = Path(args.data_dir) / args.split
    if not split_dir.exists():
        raise SystemExit(f"missing split dir: {split_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"split": args.split, "data_dir": args.data_dir}

    # --- YOLO ---
    if Path(args.yolo_weights).exists():
        print(f"[YOLO] loading {args.yolo_weights}")
        yolo = _load_yolo(args.yolo_weights)
        y_scores, y_labels = collect_pairs(
            lambda img, c: _yolo_predict(yolo, img, c), split_dir, args.conf_floor
        )
        if y_scores.size:
            before = _score_block("yolo_raw", y_scores, y_labels)
            ts = TemperatureScaler.fit_from_confidences(y_scores, y_labels, "yolo")
            probs_cal = ts.apply_to_confidence(y_scores)
            after = _score_block("yolo_temp_scaled", probs_cal, y_labels)
            ts.save(out_dir / "calibration_yolo.pkl")
            ts.save(out_dir / "calibration_yolo.json")
            report["yolo"] = {"T": ts.temperature, "before": before, "after": after}
            print(f"[YOLO] T={ts.temperature:.4f}  ECE {before['ece']:.4f} -> {after['ece']:.4f}")
        else:
            report["yolo"] = {"skipped": "no predictions collected"}
    else:
        print(f"[YOLO] weights missing, skipping")

    # --- RF-DETR ---
    if Path(args.rfdetr_weights).exists():
        print(f"[RF-DETR] loading {args.rfdetr_weights}")
        rf = _load_rfdetr(args.rfdetr_weights)
        r_scores, r_labels = collect_pairs(
            lambda img, c: _rfdetr_predict(rf, img, c), split_dir, args.conf_floor
        )
        if r_scores.size:
            before = _score_block("rfdetr_raw", r_scores, r_labels)
            ts = TemperatureScaler.fit_from_confidences(r_scores, r_labels, "rfdetr")
            probs_cal = ts.apply_to_confidence(r_scores)
            after = _score_block("rfdetr_temp_scaled", probs_cal, r_labels)
            ts.save(out_dir / "calibration_rfdetr.pkl")
            ts.save(out_dir / "calibration_rfdetr.json")
            report["rfdetr"] = {"T": ts.temperature, "before": before, "after": after}
            print(f"[RF-DETR] T={ts.temperature:.4f}  ECE {before['ece']:.4f} -> {after['ece']:.4f}")
        else:
            report["rfdetr"] = {"skipped": "no predictions collected"}
    else:
        print(f"[RF-DETR] weights missing, skipping")

    # --- Post-fusion isotonic (requires both models) ---
    if all(k in report and "T" in report[k] for k in ("yolo", "rfdetr")):
        print("[fusion] collecting WBF+shadow pairs...")
        ens = GallstoneEnsemble([yolo, rf], weights=[0.5, 0.5])
        shadow = ShadowAnalyzer()
        fused_scores, fused_labels = [], []
        for _stem, img, gt in tqdm(list(iter_split(split_dir)), desc="fusion"):
            try:
                boxes, scores, _ = ens.predict(img, conf_threshold=args.conf_floor)
            except Exception as e:
                print(f"  skip {_stem}: {e}")
                continue
            boxes = np.asarray(boxes, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            if boxes.size == 0:
                continue
            h, w = img.shape[:2]
            if boxes.max() <= 1.05:
                boxes = boxes * np.array([w, h, w, h], dtype=np.float32)
            scores = shadow.apply_to_ensemble(img, boxes, scores)
            labels = _match_and_label(boxes, scores, gt)
            fused_scores.append(scores)
            fused_labels.append(labels)
        if fused_scores:
            fs = np.concatenate(fused_scores)
            fl = np.concatenate(fused_labels)
            before = _score_block("fusion_raw", fs, fl)
            iso = IsotonicFusion().fit(fs, fl)
            fs_cal = iso.apply(fs)
            after = _score_block("fusion_isotonic", fs_cal, fl)
            iso.save(out_dir / "isotonic_fusion.pkl")
            report["fusion"] = {"before": before, "after": after}
            print(f"[fusion] ECE {before['ece']:.4f} -> {after['ece']:.4f}")

    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nCalibration report: {report_path}")


if __name__ == "__main__":
    main()
