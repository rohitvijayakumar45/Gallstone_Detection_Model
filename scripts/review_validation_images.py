"""
Manual Validation Image Review Script
Interactive tool to review and clean validation images
"""
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class ValidationReviewer:
    """Interactive tool for reviewing validation images"""

    def __init__(self, data_dir="dataset_final_resplit", debug_mode=False, split="val"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.img_dir = self.split_dir / "images"
        self.label_dir = self.split_dir / "labels"

        self.reviewed_images = set()
        self.bad_images = set()
        self.current_index = 0
        self.image_files = []
        self.debug_mode = debug_mode

        # Colors
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.YELLOW = (0, 255, 255)

    def load_images(self):
        """Load all validation images"""
        self.image_files = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        self.image_files.sort()

        # Load previous review state if exists
        review_file = self.data_dir / f"review_state_{self.split}.json"
        if review_file.exists():
            with open(review_file, 'r') as f:
                state = json.load(f)
                self.reviewed_images = set(state.get('reviewed', []))
                self.bad_images = set(state.get('bad', []))

        print(f"Loaded {len(self.image_files)} validation images")
        print(f"Previously reviewed: {len(self.reviewed_images)}")
        print(f"Previously marked as bad: {len(self.bad_images)}")

    def load_labels(self, img_path, img_shape):
        """Load labels for an image with proper coordinate conversion"""
        label_path = self.label_dir / f"{img_path.stem}.txt"

        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Get actual image dimensions
                        h, w = img_shape[:2]

                        # Check if polygon or bbox format
                        if len(parts) > 5:
                            # Polygon format - convert to bounding box
                            coords = np.array([float(x) for x in parts[1:]])
                            coords = coords.reshape(-1, 2)

                            # Convert normalized coordinates to pixels
                            x_min = int(np.min(coords[:, 0]) * w)
                            x_max = int(np.max(coords[:, 0]) * w)
                            y_min = int(np.min(coords[:, 1]) * h)
                            y_max = int(np.max(coords[:, 1]) * h)

                            if self.debug_mode:
                                print(f"DEBUG: Polygon box {line_num}: [{x_min}, {y_min}, {x_max}, {y_max}] (img: {w}x{h})")

                            boxes.append([x_min, y_min, x_max, y_max])
                        else:
                            # Bbox format: class x_center y_center width height (normalized)
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            # Convert normalized coordinates to pixels
                            x_center_px = int(x_center * w)
                            y_center_px = int(y_center * h)
                            width_px = int(width * w)
                            height_px = int(height * h)

                            # Convert to [x1, y1, x2, y2] format
                            x1 = x_center_px - width_px // 2
                            y1 = y_center_px - height_px // 2
                            x2 = x_center_px + width_px // 2
                            y2 = y_center_px + height_px // 2

                            if self.debug_mode:
                                print(f"DEBUG: Bbox {line_num}: norm=[{x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}] -> pix=[{x1}, {y1}, {x2}, {y2}] (img: {w}x{h})")

                            boxes.append([x1, y1, x2, y2])

        return boxes

    def draw_boxes(self, img, boxes, color=None):
        """Draw bounding boxes on image"""
        if color is None:
            color = self.GREEN

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        return img

    def draw_info(self, img, img_name, index, total, boxes, img_shape):
        """Draw information overlay on image"""
        h, w = img.shape[:2]

        # Create semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        # Image info
        text1 = f"Image {index + 1}/{total}: {img_name}"
        reviewed_status = "REVIEWED" if img_name in self.reviewed_images else "NEW"
        text2 = f"Size: {w}x{h} | Boxes: {len(boxes)} | Status: {reviewed_status} | {'BAD' if img_name in self.bad_images else 'GOOD'}"

        cv2.putText(img, text1, (10, 20), font, font_scale, self.WHITE, thickness)

        # Color based on status
        if img_name in self.bad_images:
            status_color = self.RED
        elif img_name in self.reviewed_images:
            status_color = self.YELLOW
        else:
            status_color = self.GREEN

        cv2.putText(img, text2, (10, 45), font, font_scale, status_color, thickness)

        # Show first box coordinates for debugging
        if boxes:
            box = boxes[0]
            text3 = f"Box: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
            cv2.putText(img, text3, (10, 70), font, font_scale, self.BLUE, thickness)

        # Instructions
        instructions = [
            "[←/A] Prev",
            "[→/D] Next",
            "[B] Bad",
            "[G] Good",
            "[X] Delete",
            "[S] Skip",
            "[Q] Quit",
            "[R] Reset"
        ]

        y_offset = h - 25
        for i, instruction in enumerate(instructions):
            x_pos = 10 + (i * 100)
            if x_pos < w - 10:
                cv2.putText(img, instruction, (x_pos, y_offset),
                           font, 0.4, self.WHITE, 1)

        return img

    def save_state(self):
        """Save review state"""
        state = {
            'reviewed': list(self.reviewed_images),
            'bad': list(self.bad_images),
            'timestamp': datetime.now().isoformat()
        }

        review_file = self.data_dir / f"review_state_{self.split}.json"
        with open(review_file, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Saved review state: {len(self.reviewed_images)} reviewed, {len(self.bad_images)} bad")

    def delete_image(self, img_path):
        """Delete image and its label"""
        label_path = self.label_dir / f"{img_path.stem}.txt"

        try:
            if img_path.exists():
                img_path.unlink()
                print(f"Deleted image: {img_path.name}")

            if label_path.exists():
                label_path.unlink()
                print(f"Deleted label: {label_path.name}")

            return True
        except Exception as e:
            print(f"Error deleting {img_path.name}: {e}")
            return False

    def review_images(self):
        """Interactive image review"""
        if not self.image_files:
            print("No images to review")
            return

        print("\n" + "=" * 60)
        print("VALIDATION IMAGE REVIEW")
        print("=" * 60)
        print("\nControls:")
        print("  [←] or [A] - Previous image")
        print("  [→] or [D] - Next image")
        print("  [SPACE]     - Next image")
        print("  [B]         - Mark as bad")
        print("  [G]         - Mark as good")
        print("  [X] or [DEL]- Delete bad image")
        print("  [S]         - Skip to next")
        print("  [Q]         - Quit and save")
        print("  [R]         - Reset review")
        print("\n" + "=" * 60)

        while True:
            # Check if we've gone past the end
            if self.current_index >= len(self.image_files):
                print("Reached end of images. Press [←] to go back or [Q] to quit.")
                # Wait for user input
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                elif key in [81, 2, 2424832]:  # Left arrow
                    self.current_index = len(self.image_files) - 1
                    continue
                else:
                    break

            img_path = self.image_files[self.current_index]
            img_name = img_path.name

            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load {img_name}")
                self.current_index += 1
                continue

            # Load and draw boxes
            boxes = self.load_labels(img_path, img.shape)
            img = self.draw_boxes(img, boxes)

            # Draw info overlay
            img = self.draw_info(img, img_name, self.current_index, len(self.image_files), boxes, img.shape)

            # Resize if too large
            h, w = img.shape[:2]
            if h > 800 or w > 1200:
                scale = min(800/h, 1200/w)
                img = cv2.resize(img, (int(w*scale), int(h*scale)))

            # Display image
            cv2.imshow('Validation Review', img)

            # Get user input
            key = cv2.waitKey(0) & 0xFF

            # Debug: show key code
            if self.debug_mode:
                print(f"DEBUG: Key pressed: {key} (char: {chr(key) if 32 <= key <= 126 else 'N/A'})")

            if key == ord('q'):  # Quit
                print("Quitting review...")
                break
            elif key == ord('b'):  # Mark as bad
                self.bad_images.add(img_name)
                print(f"Marked {img_name} as BAD")
            elif key == ord('g'):  # Mark as good
                if img_name in self.bad_images:
                    self.bad_images.remove(img_name)
                print(f"Marked {img_name} as GOOD")
            elif key == ord('x') or key == 127 or key == 8:  # Delete bad image (X key or Delete/Backspace)
                if img_name in self.bad_images:
                    if self.delete_image(img_path):
                        self.image_files.remove(img_path)
                        self.current_index -= 1  # Adjust for removed image
                else:
                    print(f"Image {img_name} not marked as bad. Press [B] first.")
            elif key == ord('r'):  # Reset review
                confirm = input("Reset all reviews? (y/n): ")
                if confirm.lower() == 'y':
                    self.reviewed_images.clear()
                    self.bad_images.clear()
                    self.current_index = 0
                    print("Review reset")
                continue
            elif key == ord('s'):  # Skip
                print(f"Skipped {img_name}")
            # Letter key alternatives for navigation
            elif key == ord('a'):  # 'A' for previous (alternative to left arrow)
                if self.current_index > 0:
                    self.current_index -= 1
                    print(f"Going back to image {self.current_index + 1}")
                    continue
                else:
                    print("Already at first image")
                    continue
            elif key == ord('d'):  # 'D' for next (alternative to right arrow)
                if self.current_index < len(self.image_files) - 1:
                    self.current_index += 1
                    print(f"Going to image {self.current_index + 1}")
                    continue
                else:
                    print("Already at last image")
                    continue
            # Try multiple arrow key codes for cross-platform compatibility
            elif key in [81, 2, 2424832]:  # Left arrow (various systems)
                if self.current_index > 0:
                    self.current_index -= 1
                    print(f"Going back to image {self.current_index + 1}")
                    continue  # Don't mark as reviewed, just go back
                else:
                    print("Already at first image")
                    continue
            elif key in [83, 3, 2555904, 2424834]:  # Right arrow (various systems)
                if self.current_index < len(self.image_files) - 1:
                    self.current_index += 1
                    print(f"Going to image {self.current_index + 1}")
                    continue  # Don't mark as reviewed, just go forward
                else:
                    print("Already at last image")
                    continue
            elif key == ord(' '):  # Space bar (next)
                # Just move to next, handled below
                pass
            else:
                # Show unknown key for debugging
                if key != 255:  # Ignore -1 (no key pressed)
                    print(f"Unknown key: {key} (try using SPACE, A/D, or arrow keys)")

            # Mark as reviewed (only if we're not just navigating)
            self.reviewed_images.add(img_name)
            self.current_index += 1

            # Auto-save every 10 images
            if len(self.reviewed_images) % 10 == 0:
                self.save_state()

        cv2.destroyAllWindows()

        # Final save
        self.save_state()

        # Summary
        print("\n" + "=" * 60)
        print("REVIEW SUMMARY")
        print("=" * 60)
        print(f"Total images: {len(self.image_files)}")
        print(f"Reviewed: {len(self.reviewed_images)}")
        print(f"Marked as bad: {len(self.bad_images)}")
        print(f"Remaining: {len(self.image_files) - len(self.reviewed_images)}")

        if self.bad_images:
            print(f"\nBad images:")
            for img_name in sorted(self.bad_images):
                print(f"  - {img_name}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Review validation images")
    parser.add_argument("--data_dir", type=str, default="dataset_final_resplit",
                       help="Dataset directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to review (train, val, test)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode to show coordinate conversion details")
    args = parser.parse_args()

    reviewer = ValidationReviewer(args.data_dir, debug_mode=args.debug, split=args.split)
    reviewer.load_images()
    reviewer.review_images()

if __name__ == "__main__":
    main()