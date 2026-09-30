import cv2
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path so 'src' can be found
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.models.model_factory import build_model
from src.models.ensemble import GallstoneEnsemble
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import argparse
from tqdm import tqdm

def parse_yolo_label(label_path, w, h):
    boxes = []
    labels = []
    if not label_path.exists():
        return boxes, labels
        
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if not parts: continue
            
            class_id = int(parts[0])
            if len(parts) > 5 and (len(parts)-1) % 2 == 0:
                # Polygon: convert to bbox
                coords = [float(x) for x in parts[1:]]
                xs = coords[0::2]
                ys = coords[1::2]
                x1, y1 = min(xs)*w, min(ys)*h
                x2, y2 = max(xs)*w, max(ys)*h
            else:
                # Bbox
                cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            
    return boxes, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfdetr", default="weights/rf_detr/checkpoint_best_ema.pth")
    parser.add_argument("--yolo", default="runs/detect/gallstone_detection/yolo26l_gallstone/weights/best.pt")
    parser.add_argument("--data_dir", default="data/processed/valid")
    args = parser.parse_args()

    print("Loading models...")
    models = []
    
    # Load YOLO
    if Path(args.yolo).exists():
        from ultralytics import YOLO
        models.append(YOLO(args.yolo))
        print(f"Loaded YOLO from {args.yolo}")
    else:
        print(f"YOLO weights not found: {args.yolo}")

    # Load RF-DETR
    if Path(args.rfdetr).exists():
        try:
            # Based on rfdetr usage
            from rfdetr import RFDETRSmall
            model = RFDETRSmall()
            state = torch.load(args.rfdetr, map_location='cpu')
            if 'model' in state: state = state['model']
            elif 'state_dict' in state: state = state['state_dict']
            model.load_state_dict(state, strict=False)
            model.eval()
            # Wrap it to match predict() signature
            class RFWrapper:
                def __init__(self, m): self.m = m
                def predict(self, img):
                    # dummy wrap for now, relies on actual implementation
                    return self.m.predict(img) if hasattr(self.m, 'predict') else ([], [], [])
            models.append(RFWrapper(model))
            print(f"Loaded RF-DETR from {args.rfdetr}")
        except Exception as e:
            print(f"Could not load RF-DETR natively: {e}")
    else:
        print(f"RF-DETR weights not found: {args.rfdetr}")

    if not models:
        print("No models loaded! Exiting.")
        return

    ensemble = GallstoneEnsemble(models, weights=[0.6, 0.4] if len(models)==2 else [1.0])
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    img_dir = Path(args.data_dir) / "images"
    lbl_dir = Path(args.data_dir) / "labels"
    
    if not img_dir.exists():
        print(f"Dataset dir {img_dir} not found!")
        return
        
    print(f"Evaluating ensemble on {img_dir}...")
    
    images = list(img_dir.glob("*.jpg"))
    for img_path in tqdm(images):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        # Ground truth
        gt_boxes, gt_labels = parse_yolo_label(lbl_path, w, h)
        target = [dict(
            boxes=torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.empty((0,4)),
            labels=torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.empty((0,), dtype=torch.int64)
        )]
        
        # Predictions
        # WBF returns normalized coordinates usually, let's check
        boxes, scores, labels = ensemble.predict(img)
        
        if len(boxes) > 0:
            # Denormalize if WBF returns 0-1
            if np.max(boxes) <= 1.05:
                boxes = np.array(boxes) * [w, h, w, h]
            pred = [dict(
                boxes=torch.tensor(boxes, dtype=torch.float32),
                scores=torch.tensor(scores, dtype=torch.float32),
                labels=torch.tensor(labels, dtype=torch.int64)
            )]
        else:
            pred = [dict(
                boxes=torch.empty((0,4), dtype=torch.float32),
                scores=torch.empty((0,), dtype=torch.float32),
                labels=torch.empty((0,), dtype=torch.int64)
            )]
            
        metric.update(pred, target)

    results = metric.compute()
    print("\n--- ENSEMBLE RESULTS ---")
    print(f"mAP@50:    {results['map_50'].item():.4f}")
    print(f"mAP@50-95: {results['map'].item():.4f}")
    print(f"mAR@100:   {results['mar_100'].item():.4f}")

if __name__ == "__main__":
    main()
