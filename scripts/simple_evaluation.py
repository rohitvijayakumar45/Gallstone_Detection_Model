"""
Simple evaluation script that works with current setup
"""
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def load_validation_data(data_dir="data/processed"):
    """Load validation images and labels"""
    val_dir = Path(data_dir) / "val"
    images = []
    labels = []

    img_dir = val_dir / "images"
    label_dir = val_dir / "labels"

    if not img_dir.exists():
        print(f"Validation directory not found: {img_dir}")
        return [], []

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
                            # Reshape to (N, 2) for x, y coordinates
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

    print(f"Loaded {len(images)} validation images")
    return images, labels

def compute_iou(box1, box2):
    """Compute IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union

def calculate_map_simple(predictions, targets, iou_threshold=0.5):
    """Calculate mean Average Precision at given IoU threshold"""
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0

    all_precisions = []

    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get('boxes', np.empty((0, 4)))
        pred_scores = pred.get('scores', np.empty(0))
        target_boxes = target.get('boxes', np.empty((0, 4)))

        if len(target_boxes) == 0:
            if len(pred_boxes) == 0:
                all_precisions.append(1.0)
            else:
                all_precisions.append(0.0)
            continue

        if len(pred_boxes) == 0:
            all_precisions.append(0.0)
            continue

        # Sort predictions by confidence
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        # Match predictions to targets
        matched_targets = set()
        true_positives = 0
        false_positives = 0

        for pred_idx in range(len(pred_boxes)):
            best_iou = 0.0
            best_target_idx = -1

            for target_idx in range(len(target_boxes)):
                if target_idx in matched_targets:
                    continue

                iou = compute_iou(pred_boxes[pred_idx], target_boxes[target_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_iou >= iou_threshold:
                true_positives += 1
                matched_targets.add(best_target_idx)
            else:
                false_positives += 1

        # Calculate precision and recall
        false_negatives = len(target_boxes) - len(matched_targets)

        if true_positives + false_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        # Use F1 score as a simple metric
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        all_precisions.append(f1)

    return np.mean(all_precisions)

def main():
    print("=" * 60)
    print("SIMPLE YOLO EVALUATION")
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
    val_images, val_labels = load_validation_data()

    if len(val_images) == 0:
        print("No validation data found")
        return

    # Evaluate model
    print("Evaluating model...")
    all_predictions = []
    all_targets = []

    for img, target_boxes in tqdm(zip(val_images, val_labels), total=len(val_images)):
        # Get predictions
        results = model.predict(img, verbose=False, conf=0.25)

        if len(results) > 0 and hasattr(results[0], 'boxes'):
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
        else:
            pred_boxes = np.empty((0, 4))
            pred_scores = np.empty(0)

        all_predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': np.zeros(len(pred_boxes))
        })
        all_targets.append({
            'boxes': target_boxes,
            'labels': np.zeros(len(target_boxes))
        })

    # Calculate mAP
    print("Calculating mAP...")
    map50 = calculate_map_simple(all_predictions, all_targets, iou_threshold=0.5)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"mAP50: {map50:.4f}")
    print(f"Number of validation images: {len(val_images)}")

    # Calculate additional metrics
    total_pred_boxes = sum(len(p['boxes']) for p in all_predictions)
    total_target_boxes = sum(len(t['boxes']) for t in all_targets)

    print(f"Total predictions: {total_pred_boxes}")
    print(f"Total ground truth boxes: {total_target_boxes}")

    if map50 >= 0.97:
        print("\n[SUCCESS] Target achieved! mAP50 >= 97%")
    elif map50 >= 0.90:
        print("\n[WARNING] Close to target. mAP50 >= 90%")
    else:
        print("\n[INFO] Below target. mAP50 < 90%")

if __name__ == "__main__":
    main()