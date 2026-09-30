"""Fit split-conformal box expansion + CRC recall bound.

Uses the same val split as calibration (fresh fold recommended for
tightness, but not required for validity). Emits:

  production_models/conformal_qhat.json  (SplitConformalBox)
  production_models/conformal_crc.json   (ConformalRiskController)
  runs/final_eval/conformal.json         (empirical coverage on test split)

Usage:
  python scripts/fit_conformal.py \
      --data-dir dataset_final_resplit \
      --calibration-split val --test-split test \
      --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.conformal import (
    ConformalRiskController,
    SplitConformalBox,
    fit_recall_bound,
    fit_split_conformal,
)
from src.evaluation.label_io import iter_split
from src.evaluation.shadow_analyzer import ShadowAnalyzer
from src.models.ensemble import GallstoneEnsemble

from scripts.fit_calibration import (
    _iou,
    _load_rfdetr,
    _load_yolo,
    _match_and_label,
)


def _collect_matched(ens: GallstoneEnsemble, split_dir: Path,
                     conf_floor: float, iou_thresh: float = 0.5):
    """Return (pred_norm, gt_norm) arrays of TP-matched pairs, plus per-image
    prediction records for CRC."""
    shadow = ShadowAnalyzer()
    pred_pairs = []  # normalised
    per_image = []

    for _stem, img, gt in tqdm(list(iter_split(split_dir)), desc=split_dir.name):
        try:
            boxes, scores, _ = ens.predict(img, conf_threshold=conf_floor)
        except Exception as e:
            print(f"  skip {_stem}: {e}")
            continue
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        h, w = img.shape[:2]
        if boxes.size and boxes.max() <= 1.05:
            boxes_pix = boxes * np.array([w, h, w, h], dtype=np.float32)
        else:
            boxes_pix = boxes
        if boxes_pix.size:
            scores = shadow.apply_to_ensemble(img, boxes_pix, scores)

        per_image.append(
            {"boxes": boxes_pix, "scores": scores, "gt": gt, "wh": (w, h)}
        )
        if boxes_pix.size == 0 or len(gt) == 0:
            continue

        # Greedy IoU matching pred -> gt at iou_thresh
        order = np.argsort(-scores)
        matched: set[int] = set()
        for idx in order:
            best_iou, best_j = 0.0, -1
            for j in range(len(gt)):
                if j in matched:
                    continue
                iou = _iou(boxes_pix[idx], gt[j])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                matched.add(best_j)
                pn = boxes_pix[idx] / np.array([w, h, w, h], dtype=np.float32)
                gn = gt[best_j] / np.array([w, h, w, h], dtype=np.float32)
                pred_pairs.append((pn, gn))

    if not pred_pairs:
        return np.empty((0, 4)), np.empty((0, 4)), per_image
    preds = np.stack([p for p, _ in pred_pairs])
    gts = np.stack([g for _, g in pred_pairs])
    return preds, gts, per_image


def main():
    p = argparse.ArgumentParser(description="Fit split-conformal + CRC")
    p.add_argument("--data-dir", default="dataset_final_resplit")
    p.add_argument("--calibration-split", default="val")
    p.add_argument("--test-split", default="test")
    p.add_argument("--yolo-weights", default="production_models/yolo_best.pt")
    p.add_argument("--rfdetr-weights", default="production_models/rfdetr_best.pth")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--conf-floor", type=float, default=0.01)
    p.add_argument("--out-dir", default="production_models")
    p.add_argument("--report", default="runs/final_eval/conformal.json")
    args = p.parse_args()

    for pth in (args.yolo_weights, args.rfdetr_weights):
        if not Path(pth).exists():
            raise SystemExit(f"missing weights: {pth}")

    yolo = _load_yolo(args.yolo_weights)
    rf = _load_rfdetr(args.rfdetr_weights)
    ens = GallstoneEnsemble([yolo, rf], weights=[0.5, 0.5])

    cal_dir = Path(args.data_dir) / args.calibration_split
    test_dir = Path(args.data_dir) / args.test_split

    print(f"[calibrate] on {cal_dir}")
    cal_preds, cal_gts, cal_per_image = _collect_matched(ens, cal_dir, args.conf_floor)

    if len(cal_preds) == 0:
        raise SystemExit("no matched TP pairs on calibration split; check thresholds")

    scb = fit_split_conformal(cal_preds, cal_gts, alpha=args.alpha, bonferroni=True)
    print(f"[SplitConformalBox] q=[{scb.q_x1:.4f},{scb.q_y1:.4f},"
          f"{scb.q_x2:.4f},{scb.q_y2:.4f}]  n={scb.n_samples}")
    scb.save(Path(args.out_dir) / "conformal_qhat.json")

    crc = fit_recall_bound(
        [{"boxes": p["boxes"], "scores": p["scores"], "gt": p["gt"]} for p in cal_per_image],
        alpha=args.alpha,
    )
    print(f"[CRC] lambda*={crc.lambda_star:.4f}  alpha={crc.alpha}")
    crc.save(Path(args.out_dir) / "conformal_crc.json")

    # Empirical coverage on test split
    print(f"[test] on {test_dir}")
    tst_preds, tst_gts, tst_per_image = _collect_matched(ens, test_dir, args.conf_floor)
    empirical = scb.coverage(tst_preds, tst_gts) if len(tst_preds) else 0.0
    print(f"[coverage] empirical={empirical:.4f}  target={1 - args.alpha:.4f}")

    # Empirical FNR at CRC lambda* on test split
    from src.conformal.risk_control import _fnr
    fnrs = [
        _fnr(np.asarray(p["boxes"]), np.asarray(p["scores"]),
             np.asarray(p["gt"]), crc.lambda_star, crc.iou_thresh)
        for p in tst_per_image
    ]
    mean_fnr = float(np.mean(fnrs)) if fnrs else 0.0
    print(f"[FNR] mean={mean_fnr:.4f}  target<={args.alpha}")

    report = {
        "alpha": args.alpha,
        "split_conformal": {
            "q": [scb.q_x1, scb.q_y1, scb.q_x2, scb.q_y2],
            "n_calibration": scb.n_samples,
            "bonferroni": scb.bonferroni,
            "empirical_coverage_test": empirical,
            "target_coverage": 1.0 - args.alpha,
            "n_test_pairs": int(len(tst_preds)),
        },
        "crc": {
            "lambda_star": crc.lambda_star,
            "n_calibration_images": crc.n_images,
            "empirical_fnr_test": mean_fnr,
            "target_fnr": args.alpha,
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
