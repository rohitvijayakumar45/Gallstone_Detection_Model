"""
Test script to verify bounding box coordinate conversion
"""
import cv2
import numpy as np
from pathlib import Path

def test_coordinate_conversion():
    """Test coordinate conversion on a few sample images"""

    data_dir = Path("dataset_final_resplit")
    val_dir = data_dir / "val"
    img_dir = val_dir / "images"
    label_dir = val_dir / "labels"

    # Get first few images
    image_files = list(img_dir.glob("*.jpg"))[:3]

    print("=" * 60)
    print("COORDINATE CONVERSION TEST")
    print("=" * 60)

    for img_path in image_files:
        print(f"\nTesting: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image")
            continue

        h, w = img.shape[:2]
        print(f"Image size: {w}x{h}")

        # Load labels
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"No label file found")
            continue

        print(f"Label file: {label_path.name}")

        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    print(f"\n  Line {line_num + 1}: {line[:80]}...")

                    # Check format
                    if len(parts) > 5:
                        print(f"  Format: POLYGON ({len(parts)-1} coordinates)")
                        coords = np.array([float(x) for x in parts[1:]])
                        coords = coords.reshape(-1, 2)

                        # Show first few coordinates
                        print(f"  First 3 coords (normalized): {coords[:3].tolist()}")

                        # Convert to pixels
                        x_min = int(np.min(coords[:, 0]) * w)
                        x_max = int(np.max(coords[:, 0]) * w)
                        y_min = int(np.min(coords[:, 1]) * h)
                        y_max = int(np.max(coords[:, 1]) * h)

                        print(f"  Bounding box (pixels): [{x_min}, {y_min}, {x_max}, {y_max}]")

                        # Check if box is within image bounds
                        if x_min < 0 or x_max > w or y_min < 0 or y_max > h:
                            print(f"  WARNING: Box extends outside image bounds!")

                    else:
                        print(f"  Format: BBOX")
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        print(f"  Class: {class_id}")
                        print(f"  Center (normalized): [{x_center:.3f}, {y_center:.3f}]")
                        print(f"  Size (normalized): [{width:.3f}, {height:.3f}]")

                        # Convert to pixels
                        x_center_px = int(x_center * w)
                        y_center_px = int(y_center * h)
                        width_px = int(width * w)
                        height_px = int(height * h)

                        print(f"  Center (pixels): [{x_center_px}, {y_center_px}]")
                        print(f"  Size (pixels): [{width_px}, {height_px}]")

                        # Convert to [x1, y1, x2, y2]
                        x1 = x_center_px - width_px // 2
                        y1 = y_center_px - height_px // 2
                        x2 = x_center_px + width_px // 2
                        y2 = y_center_px + height_px // 2

                        print(f"  Bounding box (pixels): [{x1}, {y1}, {x2}, {y2}]")

                        # Check if box is within image bounds
                        if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
                            print(f"  WARNING: Box extends outside image bounds!")

                        # Check if box is reasonable size
                        box_width = x2 - x1
                        box_height = y2 - y1
                        if box_width < 10 or box_height < 10:
                            print(f"  WARNING: Box is very small: {box_width}x{box_height}")

        # Create visualization
        print(f"\n  Creating visualization...")

        # Reload labels for visualization
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        if len(parts) > 5:
                            # Polygon
                            coords = np.array([float(x) for x in parts[1:]])
                            coords = coords.reshape(-1, 2)
                            x_min = int(np.min(coords[:, 0]) * w)
                            x_max = int(np.max(coords[:, 0]) * w)
                            y_min = int(np.min(coords[:, 1]) * h)
                            y_max = int(np.max(coords[:, 1]) * h)
                            boxes.append([x_min, y_min, x_max, y_max])
                        else:
                            # Bbox
                            x_center, y_center, width, height = map(float, parts[1:5])
                            x_center_px = int(x_center * w)
                            y_center_px = int(y_center * h)
                            width_px = int(width * w)
                            height_px = int(height * h)
                            x1 = x_center_px - width_px // 2
                            y1 = y_center_px - height_px // 2
                            x2 = x_center_px + width_px // 2
                            y2 = y_center_px + height_px // 2
                            boxes.append([x1, y1, x2, y2])

        # Draw boxes
        vis_img = img.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text
        cv2.putText(vis_img, f"Image: {img_path.name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, f"Size: {w}x{h}, Boxes: {len(boxes)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save visualization
        output_path = Path(f"test_output_{img_path.stem}.jpg")
        cv2.imwrite(str(output_path), vis_img)
        print(f"  Saved visualization to: {output_path}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nCheck the generated test_output_*.jpg files to verify")
    print("that the green boxes match the gallstones in the images.")

if __name__ == "__main__":
    test_coordinate_conversion()