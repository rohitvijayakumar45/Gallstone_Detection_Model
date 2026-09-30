"""
Debug script to check model predictions and labels
"""
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

def load_validation_data(data_dir="data/processed"):
    """Load validation images and labels"""
    val_dir = Path(data_dir) / "val"
    images = []
    labels = []
    image_names = []

    img_dir = val_dir / "images"
    label_dir = val_dir / "labels"

    if not img_dir.exists():
        print(f"Validation directory not found: {img_dir}")
        return [], [], []

    for img_path in img_dir.glob("*.jpg"):
        label_path = label_dir / f"{img_path.stem}.txt"

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load labels
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Check if it's polygon format (many coordinates) or bbox format (5 coordinates)
                        if len(parts) > 5:
                            # Polygon format - convert to bounding box
                            coords = np.array([float(x) for x in parts[1:]])
                            coords = coords.reshape(-1, 2)

                            # Get bounding box from polygon
                            x_min = np.min(coords[:, 0])
                            x_max = np.max(coords[:, 0])
                            y_min = np.min(coords[:, 1])
                            y_max = np.max(coords[:, 1])

                            # Convert to pixel coordinates
                            h, w = img.shape[:2]
                            x1 = int(x_min * w)
                            y1 = int(y_min * h)
                            x2 = int(x_max * w)
                            y2 = int(y_max * h)
                            boxes.append([x1, y1, x2, y2])
                        else:
                            # Bbox format: class x_center y_center width height (normalized)
                            x_center, y_center, width, height = map(float, parts[1:5])
                            h, w = img.shape[:2]

                            # Convert to pixel coordinates [x1, y1, x2, y2]
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            boxes.append([x1, y1, x2, y2])

        images.append(img)
        labels.append(np.array(boxes) if len(boxes) > 0 else np.empty((0, 4)))
        image_names.append(img_path.name)

    print(f"Loaded {len(images)} validation images")
    return images, labels, image_names

def main():
    print("=" * 60)
    print("DEBUG: Model Predictions vs Ground Truth")
    print("=" * 60)

    # Load YOLO model
    print("Loading YOLO model...")
    yolo_path = "runs/detect/gallstone_detection/yolo26m_gallstone/weights/best.pt"
    if not Path(yolo_path).exists():
        print(f"YOLO model not found at {yolo_path}")
        return

    model = YOLO(yolo_path)
    print(f"Loaded YOLO model from {yolo_path}")

    # Load validation data
    print("Loading validation data...")
    val_images, val_labels, image_names = load_validation_data()

    if len(val_images) == 0:
        print("No validation data found")
        return

    # Check first few examples
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    for i in range(min(5, len(val_images))):
        img = val_images[i]
        target_boxes = val_labels[i]
        img_name = image_names[i]

        print(f"\nImage {i+1}: {img_name}")
        print(f"Ground truth boxes: {len(target_boxes)}")

        if len(target_boxes) > 0:
            print(f"  GT boxes (pixels): {target_boxes.tolist()}")

        # Get predictions
        results = model.predict(img, verbose=False, conf=0.25)

        if len(results) > 0 and hasattr(results[0], 'boxes'):
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            print(f"Predictions: {len(pred_boxes)}")

            if len(pred_boxes) > 0:
                print(f"  Prediction boxes (pixels): {pred_boxes.tolist()}")
                print(f"  Prediction scores: {pred_scores.tolist()}")

                # Check if any predictions match ground truth
                if len(target_boxes) > 0:
                    for pred_box, pred_score in zip(pred_boxes, pred_scores):
                        for target_box in target_boxes:
                            # Simple IoU calculation
                            x1 = max(pred_box[0], target_box[0])
                            y1 = max(pred_box[1], target_box[1])
                            x2 = min(pred_box[2], target_box[2])
                            y2 = min(pred_box[3], target_box[3])

                            if x2 > x1 and y2 > y1:
                                intersection = (x2 - x1) * (y2 - y1)
                                area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                                area_target = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                                union = area_pred + area_target - intersection

                                if union > 0:
                                    iou = intersection / union
                                    if iou > 0.5:
                                        print(f"  MATCH! IoU: {iou:.3f}, Score: {pred_score:.3f}")
        else:
            print("No predictions")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    total_gt_boxes = sum(len(labels) for labels in val_labels)
    total_pred_boxes = 0
    total_high_conf_pred = 0

    for img in val_images:
        results = model.predict(img, verbose=False, conf=0.25)
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            total_pred_boxes += len(pred_boxes)
            total_high_conf_pred += np.sum(pred_scores > 0.5)

    print(f"Total ground truth boxes: {total_gt_boxes}")
    print(f"Total predictions (conf > 0.25): {total_pred_boxes}")
    print(f"Total high confidence predictions (conf > 0.5): {total_high_conf_pred}")
    print(f"Predictions per GT box: {total_pred_boxes / max(total_gt_boxes, 1):.2f}")

if __name__ == "__main__":
    main()