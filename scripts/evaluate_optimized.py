import cv2
import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.models.ensemble import GallstoneEnsemble
from src.evaluation.shadow_analyzer import ShadowAnalyzer
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def parse_yolo_label(label_path, w, h):
    boxes, labels = [], []
    if not label_path.exists(): return boxes, labels
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts: continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) >= 6: # Polygon
                xs, ys = coords[0::2], coords[1::2]
                x1, y1, x2, y2 = min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h
            else: # Bbox
                cx, cy, bw, bh = coords
                x1, y1 = (cx - bw/2)*w, (cy - bh/2)*h
                x2, y2 = (cx + bw/2)*w, (cy + bh/2)*h
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
    return boxes, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", default="runs/detect/gallstone_detection_hpo/hpo_trial_15/weights/best.pt")
    parser.add_argument("--data_dir", default="data/processed/valid")
    parser.add_argument("--multi_scale", action="store_true", default=True)
    parser.add_argument("--use_shadow", action="store_true", default=True)
    args = parser.parse_args()

    print("Loading optimized ensemble...")
    models = []
    if Path(args.yolo).exists():
        models.append(YOLO(args.yolo))
        
    ensemble = GallstoneEnsemble(models)
    shadow_analyzer = ShadowAnalyzer()
    metric = MeanAveragePrecision(box_format='xyxy')
    
    img_dir = Path(args.data_dir) / "images"
    lbl_dir = Path(args.data_dir) / "labels"
    
    images = list(img_dir.glob("*.jpg"))
    print(f"Evaluating {len(images)} images with Multi-Scale & Shadow Verification...")
    
    for img_path in tqdm(images):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        # Ground Truth
        gt_boxes, gt_labels = parse_yolo_label(lbl_dir / f"{img_path.stem}.txt", w, h)
        target = [dict(
            boxes=torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty((0,4)),
            labels=torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.empty((0,), dtype=torch.int64)
        )]
        
        # Optimized Prediction
        if args.multi_scale:
            boxes, scores, labels = ensemble.predict_multi_scale(img, scales=[640, 800, 1024])
        else:
            boxes, scores, labels = ensemble.predict(img)
            
        if len(boxes) > 0:
            # Denormalize
            boxes = np.array(boxes) * [w, h, w, h]
            
            # Clinical Shadow Verification
            if args.use_shadow:
                scores = shadow_analyzer.apply_to_ensemble(img, boxes, scores)
                
            # Filter low confidence after shadow check
            mask = scores > 0.25
            boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
            
            pred = [dict(
                boxes=torch.tensor(boxes, dtype=torch.float32) if len(boxes)>0 else torch.empty((0,4)),
                scores=torch.tensor(scores, dtype=torch.float32) if len(scores)>0 else torch.empty((0,)),
                labels=torch.tensor(labels, dtype=torch.int64) if len(labels)>0 else torch.empty((0,), dtype=torch.int64)
            )]
        else:
            pred = [dict(
                boxes=torch.empty((0,4)),
                scores=torch.empty((0,)),
                labels=torch.empty((0,), dtype=torch.int64)
            )]
            
        metric.update(pred, target)

    results = metric.compute()
    print("\n--- OPTIMIZED ENSEMBLE RESULTS ---")
    print(f"mAP@50:    {results['map_50'].item():.4f}")
    print(f"mAP@50-95: {results['map'].item():.4f}")

if __name__ == "__main__":
    main()
