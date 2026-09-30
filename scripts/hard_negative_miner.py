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
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", default="runs/detect/gallstone_detection_hpo/hpo_trial_15/weights/best.pt")
    parser.add_argument("--rfdetr", default="weights/rf_detr/checkpoint_best_ema.pth")
    parser.add_argument("--data_dir", default="data/processed/train")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--output", default="hard_negatives.txt")
    args = parser.parse_args()

    print("Loading models...")
    models = []
    if Path(args.yolo).exists():
        models.append(YOLO(args.yolo))
    if Path(args.rfdetr).exists():
        from src.models.model_factory import build_model
        rf_detr = build_model("rf_detr", {"model_size": "small"})
        try:
            state = torch.load(args.rfdetr, map_location='cpu')
            if 'model' in state: state = state['model']
            rf_detr.model.load_state_dict(state, strict=False)
            models.append(rf_detr.model)
        except Exception as e:
            print(f"RF-DETR load failed: {e}")

    ensemble = GallstoneEnsemble(models)
    
    img_dir = Path(args.data_dir) / "images"
    lbl_dir = Path(args.data_dir) / "labels"
    
    hard_negatives = []
    print(f"Scanning {img_dir} for hard negatives...")
    
    images = list(img_dir.glob("*.jpg"))
    for img_path in tqdm(images):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        
        # Check if ground truth is empty
        has_gt = False
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            has_gt = True
            
        if has_gt: continue # Only looking for False Positives in empty images
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        boxes, scores, labels = ensemble.predict(img, conf_threshold=args.conf)
        
        if len(boxes) > 0:
            max_score = np.max(scores)
            hard_negatives.append((img_path.name, max_score))
            
    # Sort by confidence
    hard_negatives.sort(key=lambda x: x[1], reverse=True)
    
    with open(args.output, 'w') as f:
        for name, score in hard_negatives:
            f.write(f"{name} {score:.4f}\n")
            
    print(f"\nFound {len(hard_negatives)} potential hard negatives. Saved to {args.output}")
    print("These are images with NO labels where the model found a stone. Check them in clean_dataset.py!")

if __name__ == "__main__":
    main()
