from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@dataclass
class ImageRecord:
    image_path: Path
    label_path: Path
    width: int
    height: int
    gt_boxes: np.ndarray
    gt_labels: np.ndarray


def parse_yolo_label(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
        if len(vals) == 4:
            cx, cy, bw, bh = vals
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            x2 = (cx + bw / 2) * width
            y2 = (cy + bh / 2) * height
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            xs = vals[0::2]
            ys = vals[1::2]
            x1, y1 = min(xs) * width, min(ys) * height
            x2, y2 = max(xs) * width, max(ys) * height
        else:
            continue
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_records(data_dir: Path, split: str) -> list[ImageRecord]:
    split_dir = data_dir / split
    img_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {img_dir}")
    records: list[ImageRecord] = []
    for image_path in sorted(img_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes, labels = parse_yolo_label(label_path, w, h)
        records.append(ImageRecord(image_path, label_path, w, h, boxes, labels))
    return records


def valid_box_mask(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    boxes = boxes.astype(np.float32)
    return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])


def clamp_norm_prediction(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, 1.0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, 1.0)
    keep = valid_box_mask(boxes)
    return boxes[keep], scores.astype(np.float32)[keep], labels.astype(np.int64)[keep]


def yolo_predict(model, image_bgr: np.ndarray, conf: float, augment: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    results = model.predict(image_bgr, conf=conf, iou=0.7, max_det=300, verbose=False, augment=augment)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    r = results[0].boxes
    return clamp_norm_prediction(
        r.xyxyn.detach().cpu().numpy().astype(np.float32),
        r.conf.detach().cpu().numpy().astype(np.float32),
        r.cls.detach().cpu().numpy().astype(np.int64),
    )


def rfdetr_predict(model, image_bgr: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    det = model.predict(rgb, threshold=conf)
    if len(det) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    boxes = det.xyxy.astype(np.float32) / np.asarray([w, h, w, h], dtype=np.float32)
    scores = det.confidence.astype(np.float32)
    labels = det.class_id.astype(np.int64)
    return clamp_norm_prediction(boxes, scores, labels)


def hflip_boxes(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    out = boxes.copy()
    out[:, 0] = 1.0 - boxes[:, 2]
    out[:, 2] = 1.0 - boxes[:, 0]
    return out


def concat_preds(preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = [p[0] for p in preds if len(p[0])]
    scores = [p[1] for p in preds if len(p[1])]
    labels = [p[2] for p in preds if len(p[2])]
    if not boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(labels)


def cached_arrays(preds: dict, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(preds[name]["boxes"], dtype=np.float32)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
    else:
        boxes = boxes.reshape(-1, 4)
    return (
        boxes,
        np.asarray(preds[name]["scores"], dtype=np.float32),
        np.asarray(preds[name]["labels"], dtype=np.int64),
    )


def filter_conf(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, conf: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(scores) == 0:
        return boxes, scores, labels
    keep = scores >= conf
    return boxes[keep], scores[keep], labels[keep]


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thr: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores, labels
    keep_all: list[int] = []
    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0]
        order = idxs[np.argsort(scores[idxs])[::-1]]
        while len(order):
            cur = int(order[0])
            keep_all.append(cur)
            if len(order) == 1:
                break
            rest = order[1:]
            order = rest[box_iou_one_to_many(boxes[cur], boxes[rest]) < iou_thr]
    keep = np.asarray(keep_all, dtype=np.int64)
    keep = keep[np.argsort(scores[keep])[::-1]]
    return boxes[keep], scores[keep], labels[keep]


def build_cache(args, records: list[ImageRecord]) -> dict[str, dict[str, dict[str, list]]]:
    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild_cache:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    from ultralytics import YOLO
    from rfdetr import RFDETRSmall

    print("Load YOLO:", args.yolo)
    yolo = YOLO(args.yolo)
    print("Load RF-DETR:", args.rfdetr)
    rfdetr = RFDETRSmall(pretrain_weights=args.rfdetr)
    try:
        rfdetr.optimize_for_inference()
    except Exception:
        pass

    cache: dict[str, dict[str, dict[str, list]]] = {}
    for rec in tqdm(records, desc="cache predictions"):
        image = cv2.imread(str(rec.image_path))
        if image is None:
            continue
        yolo_preds = [yolo_predict(yolo, image, args.model_conf, args.yolo_augment)]
        rfdetr_preds = [rfdetr_predict(rfdetr, image, args.model_conf)]
        if args.tta_hflip:
            flip = cv2.flip(image, 1)
            yb, ys, yl = yolo_predict(yolo, flip, args.model_conf, args.yolo_augment)
            rb, rs, rl = rfdetr_predict(rfdetr, flip, args.model_conf)
            yolo_preds.append((hflip_boxes(yb), ys, yl))
            rfdetr_preds.append((hflip_boxes(rb), rs, rl))
        yb, ys, yl = concat_preds(yolo_preds)
        rb, rs, rl = concat_preds(rfdetr_preds)
        cache[rec.image_path.name] = {
            "yolo": {"boxes": yb.tolist(), "scores": ys.tolist(), "labels": yl.tolist()},
            "rfdetr": {"boxes": rb.tolist(), "scores": rs.tolist(), "labels": rl.tolist()},
        }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def fuse_wbf(preds: dict, yolo_weight: float, conf: float, iou: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ensemble_boxes import weighted_boxes_fusion

    if yolo_weight <= 0.0:
        return filter_conf(*cached_arrays(preds, "rfdetr"), conf)
    if yolo_weight >= 1.0:
        return filter_conf(*cached_arrays(preds, "yolo"), conf)

    model_boxes = [preds["rfdetr"]["boxes"], preds["yolo"]["boxes"]]
    model_scores = [preds["rfdetr"]["scores"], preds["yolo"]["scores"]]
    model_labels = [preds["rfdetr"]["labels"], preds["yolo"]["labels"]]
    weights = [1.0 - yolo_weight, yolo_weight]
    boxes, scores, labels = weighted_boxes_fusion(
        model_boxes,
        model_scores,
        model_labels,
        weights=weights,
        iou_thr=iou,
        skip_box_thr=conf,
        conf_type="avg",
        allows_overflow=False,
    )
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def fuse_rf_primary_union(
    preds: dict, rf_conf: float, yolo_conf: float, add_iou: float, nms_iou: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rb, rs, rl = filter_conf(*cached_arrays(preds, "rfdetr"), rf_conf)
    yb, ys, yl = filter_conf(*cached_arrays(preds, "yolo"), yolo_conf)
    if len(yb):
        add = np.ones((len(yb),), dtype=bool)
        for i, box in enumerate(yb):
            same_cls = rb[rl == yl[i]] if len(rb) else rb
            if len(same_cls) and float(box_iou_one_to_many(box, same_cls).max()) >= add_iou:
                add[i] = False
        boxes = np.concatenate([rb, yb[add]]) if len(rb) else yb[add]
        scores = np.concatenate([rs, ys[add]]) if len(rs) else ys[add]
        labels = np.concatenate([rl, yl[add]]) if len(rl) else yl[add]
    else:
        boxes, scores, labels = rb, rs, rl
    return nms(boxes, scores, labels, nms_iou)


def fuse_one(
    preds: dict,
    fusion: str,
    yolo_weight: float,
    conf: float,
    iou: float,
    yolo_conf: float,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fusion == "wbf":
        return fuse_wbf(preds, yolo_weight, conf, iou)
    if fusion == "rfdetr":
        return nms(*filter_conf(*cached_arrays(preds, "rfdetr"), conf), nms_iou)
    if fusion == "yolo":
        return nms(*filter_conf(*cached_arrays(preds, "yolo"), conf), nms_iou)
    if fusion == "rf_primary_union":
        return fuse_rf_primary_union(preds, conf, yolo_conf, iou, nms_iou)
    raise ValueError(f"Unknown fusion: {fusion}")


def eval_config(
    records: list[ImageRecord],
    cache: dict,
    fusion: str,
    yolo_weight: float,
    conf: float,
    iou: float,
    yolo_conf: float,
    nms_iou: float,
) -> dict[str, float]:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", max_detection_thresholds=[1, 10, 300])
    for rec in records:
        boxes, scores, labels = fuse_one(cache[rec.image_path.name], fusion, yolo_weight, conf, iou, yolo_conf, nms_iou)
        if len(boxes):
            pred_boxes = boxes * np.asarray([rec.width, rec.height, rec.width, rec.height], dtype=np.float32)
        else:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
        preds = [
            {
                "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                "scores": torch.tensor(scores, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        ]
        targets = [
            {
                "boxes": torch.tensor(rec.gt_boxes, dtype=torch.float32),
                "labels": torch.tensor(rec.gt_labels, dtype=torch.int64),
            }
        ]
        metric.update(preds, targets)
    res = metric.compute()
    return {
        "map50": float(res["map_50"]),
        "map50_95": float(res["map"]),
        "mar300": float(res["mar_300"]),
        "yolo_weight": yolo_weight,
        "rfdetr_weight": 1.0 - yolo_weight,
        "conf": conf,
        "yolo_conf": yolo_conf,
        "wbf_iou": iou,
        "fusion_iou": iou,
        "nms_iou": nms_iou,
        "fusion": fusion,
    }


def sweep(records: list[ImageRecord], cache: dict, args) -> list[dict[str, float]]:
    yolo_weights = np.arange(args.yolo_weight_min, args.yolo_weight_max + 1e-9, args.yolo_weight_step)
    confs = np.arange(args.conf_min, args.conf_max + 1e-9, args.conf_step)
    ious = np.arange(args.iou_min, args.iou_max + 1e-9, args.iou_step)
    yolo_confs = np.arange(args.yolo_conf_min, args.yolo_conf_max + 1e-9, args.yolo_conf_step)
    results: list[dict[str, float]] = []
    if args.fusion == "wbf":
        total = len(yolo_weights) * len(confs) * len(ious)
    elif args.fusion == "rf_primary_union":
        total = len(confs) * len(yolo_confs) * len(ious)
    else:
        total = len(confs)
    with tqdm(total=total, desc=f"sweep {args.fusion}") as bar:
        if args.fusion == "wbf":
            for yw in yolo_weights:
                for conf in confs:
                    for iou in ious:
                        out = eval_config(records, cache, args.fusion, float(yw), float(conf), float(iou), float(conf), args.nms_iou)
                        results.append(out)
                        bar.update(1)
        elif args.fusion == "rf_primary_union":
            for conf in confs:
                for yolo_conf in yolo_confs:
                    for iou in ious:
                        out = eval_config(records, cache, args.fusion, 0.0, float(conf), float(iou), float(yolo_conf), args.nms_iou)
                        results.append(out)
                        bar.update(1)
        else:
            for conf in confs:
                out = eval_config(records, cache, args.fusion, 0.0, float(conf), 0.5, float(conf), args.nms_iou)
                results.append(out)
                bar.update(1)
    results.sort(key=lambda x: (x["map50"], x["map50_95"]), reverse=True)
    return results


def oracle_recall(records: list[ImageRecord], cache: dict, conf: float, iou_thr: float) -> dict[str, float]:
    hits = {"rfdetr": 0, "yolo": 0, "either": 0}
    total = 0
    for rec in records:
        pred = cache[rec.image_path.name]
        rb, rs, rl = filter_conf(*cached_arrays(pred, "rfdetr"), conf)
        yb, ys, yl = filter_conf(*cached_arrays(pred, "yolo"), conf)
        rb_px = rb * np.asarray([rec.width, rec.height, rec.width, rec.height], dtype=np.float32)
        yb_px = yb * np.asarray([rec.width, rec.height, rec.width, rec.height], dtype=np.float32)
        for gt, glabel in zip(rec.gt_boxes, rec.gt_labels):
            total += 1
            rf_hit = len(rb_px[rl == glabel]) > 0 and float(box_iou_one_to_many(gt, rb_px[rl == glabel]).max()) >= iou_thr
            yolo_hit = len(yb_px[yl == glabel]) > 0 and float(box_iou_one_to_many(gt, yb_px[yl == glabel]).max()) >= iou_thr
            hits["rfdetr"] += int(rf_hit)
            hits["yolo"] += int(yolo_hit)
            hits["either"] += int(rf_hit or yolo_hit)
    return {
        "oracle_conf": conf,
        "oracle_iou": iou_thr,
        "gt": total,
        "rfdetr_recall": hits["rfdetr"] / max(total, 1),
        "yolo_recall": hits["yolo"] / max(total, 1),
        "either_recall": hits["either"] / max(total, 1),
        "yolo_extra_gt_hits": (hits["either"] - hits["rfdetr"]) / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", default="valid", choices=["valid", "val", "test", "train"])
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--rfdetr", required=True)
    parser.add_argument("--cache", default="runs/detect/ensemble_cache/preds_valid.json")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--model-conf", type=float, default=0.001)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--yolo-augment", action="store_true")
    parser.add_argument("--yolo-weight-min", type=float, default=0.1)
    parser.add_argument("--yolo-weight-max", type=float, default=0.7)
    parser.add_argument("--yolo-weight-step", type=float, default=0.05)
    parser.add_argument("--conf-min", type=float, default=0.001)
    parser.add_argument("--conf-max", type=float, default=0.35)
    parser.add_argument("--conf-step", type=float, default=0.025)
    parser.add_argument("--iou-min", type=float, default=0.35)
    parser.add_argument("--iou-max", type=float, default=0.75)
    parser.add_argument("--iou-step", type=float, default=0.05)
    parser.add_argument("--output", default="runs/detect/ensemble_sweep_valid.json")
    parser.add_argument("--fusion", choices=["wbf", "rfdetr", "yolo", "rf_primary_union"], default="wbf")
    parser.add_argument("--yolo-conf-min", type=float, default=0.05)
    parser.add_argument("--yolo-conf-max", type=float, default=0.50)
    parser.add_argument("--yolo-conf-step", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--oracle-conf", type=float, default=0.25)
    parser.add_argument("--oracle-iou", type=float, default=0.50)
    args = parser.parse_args()

    records = load_records(Path(args.data_dir), args.split)
    print(f"records={len(records)} gt={sum(len(r.gt_boxes) for r in records)} split={args.split}")
    cache = build_cache(args, records)
    results = sweep(records, cache, args)
    oracle = oracle_recall(records, cache, args.oracle_conf, args.oracle_iou)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"best": results[0], "top20": results[:20], "count": len(results), "oracle": oracle}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nORACLE")
    print(json.dumps(oracle, indent=2))
    print("\nBEST")
    print(json.dumps(results[0], indent=2))
    print("\nTOP 5")
    for row in results[:5]:
        print(row)
    print(f"\nsaved={output}")


if __name__ == "__main__":
    main()
