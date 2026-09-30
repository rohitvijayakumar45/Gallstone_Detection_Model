import cv2
import numpy as np
import os
import argparse
from pathlib import Path

class DatasetCleaner:
    def __init__(self, data_dir, delete_processed=False):
        self.data_dir = Path(data_dir)
        self.delete_processed = delete_processed
        self.processed_dir = Path("data/processed")
        
        self.image_paths = self._get_image_paths()
        self.total_images = len(self.image_paths)
        self.current_idx = 0
        self.deleted_count = 0
        
        self.mode = "VIEW" # VIEW, CROP, MASK
        
        # State variables for current image
        self.img_path = None
        self.label_path = None
        self.img = None          # The clean, original image for saving
        self.display_img = None  # The image with overlays drawn on it
        self.h = 0
        self.w = 0
        
        # YOLO normalized polygons: list of (class_id, [x1, y1, x2, y2, ...])
        self.polygons = []
        
        # Crop state
        self.crop_start = None
        self.crop_end = None
        self.is_cropping = False
        
        # Mask state
        self.current_mask_pts = []
        
        cv2.namedWindow("Dataset Editor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Dataset Editor", self.mouse_callback)

    def _get_image_paths(self):
        splits = ["train", "val", "test", "valid"]
        paths = []
        for split in splits:
            img_dir = self.data_dir / split / "images"
            if img_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
                    paths.extend(list(img_dir.glob(ext)))
        return paths

    def _load_labels(self):
        self.polygons = []
        if not self.label_path.exists():
            return
            
        with open(self.label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    self.polygons.append((class_id, coords))
                elif len(parts) == 5:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:]]
                    x1, y1 = cx - bw/2, cy - bh/2
                    x2, y2 = cx + bw/2, cy + bh/2
                    self.polygons.append((class_id, [x1, y1, x2, y1, x2, y2, x1, y2]))

    def _save_labels(self):
        with open(self.label_path, 'w') as f:
            for class_id, coords in self.polygons:
                # clip coordinates to 0-1
                coords = [max(0.0, min(1.0, c)) for c in coords]
                coord_str = " ".join([f"{c:.6f}" for c in coords])
                f.write(f"{class_id} {coord_str}\n")

    def _refresh_display(self):
        if self.img is None:
            return
            
        self.display_img = self.img.copy()
        
        # Draw existing polygons
        for class_id, coords in self.polygons:
            pts = []
            for i in range(0, len(coords), 2):
                pts.append([int(coords[i] * self.w), int(coords[i+1] * self.h)])
            pts = np.array(pts, np.int32)
            
            overlay = self.display_img.copy()
            cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
            cv2.addWeighted(overlay, 0.3, self.display_img, 0.7, 0, self.display_img)
            cv2.polylines(self.display_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
        # Draw current crop box
        if self.mode == "CROP" and self.crop_start and self.crop_end:
            cv2.rectangle(self.display_img, self.crop_start, self.crop_end, (0, 0, 255), 2)
            
        # Draw current mask points
        if self.mode == "MASK" and len(self.current_mask_pts) > 0:
            for pt in self.current_mask_pts:
                cv2.circle(self.display_img, pt, 3, (255, 0, 0), -1)
            if len(self.current_mask_pts) > 1:
                pts = np.array(self.current_mask_pts, np.int32)
                cv2.polylines(self.display_img, [pts], isClosed=False, color=(255, 0, 0), thickness=2)
                
        # Draw UI
        overlay = self.display_img.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, self.display_img, 0.4, 0, self.display_img)
        
        cv2.putText(self.display_img, f"Image: {self.img_path.name} ({self.current_idx+1}/{self.total_images}) | MODE: {self.mode}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.mode == "VIEW":
            cmds = "[Space/N]: Next | [B]: Back | [D]: Delete | [C]: Crop | [M]: Mask | [Q]: Quit"
        elif self.mode == "CROP":
            cmds = "Drag to box. [S]: Save Crop | [V]: Cancel/View Mode"
        elif self.mode == "MASK":
            cmds = "L-Click: Add pt | R-Click: Undo pt | [S]: Save Mask | [V]: Cancel/View Mode"
            
        cv2.putText(self.display_img, cmds, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Dataset Editor", self.display_img)

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == "CROP":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.crop_start = (x, y)
                self.crop_end = (x, y)
                self.is_cropping = True
            elif event == cv2.EVENT_MOUSEMOVE and self.is_cropping:
                self.crop_end = (x, y)
                self._refresh_display()
            elif event == cv2.EVENT_LBUTTONUP:
                self.crop_end = (x, y)
                self.is_cropping = False
                self._refresh_display()
                
        elif self.mode == "MASK":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.current_mask_pts.append((x, y))
                self._refresh_display()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if len(self.current_mask_pts) > 0:
                    self.current_mask_pts.pop()
                    self._refresh_display()

    def _apply_crop(self):
        if not self.crop_start or not self.crop_end:
            return
            
        x1, y1 = self.crop_start
        x2, y2 = self.crop_end
        
        # Ensure correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Ensure within bounds and size > 0
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.w, x2), min(self.h, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            print("Crop area too small!")
            return
            
        # 1. Crop image
        self.img = self.img[y1:y2, x1:x2]
        cv2.imwrite(str(self.img_path), self.img)
        
        # 2. Adjust normalized coordinates
        crop_w = x2 - x1
        crop_h = y2 - y1
        new_polygons = []
        
        for class_id, coords in self.polygons:
            new_coords = []
            for i in range(0, len(coords), 2):
                px = coords[i] * self.w
                py = coords[i+1] * self.h
                # translate and re-normalize
                nx = (px - x1) / crop_w
                ny = (py - y1) / crop_h
                new_coords.extend([nx, ny])
            new_polygons.append((class_id, new_coords))
            
        self.polygons = new_polygons
        self._save_labels()
        
        self.h, self.w = self.img.shape[:2]
        self.crop_start = None
        self.crop_end = None
        self.mode = "VIEW"
        print("Crop applied and labels updated.")

    def _apply_mask(self):
        if len(self.current_mask_pts) < 3:
            print("Need at least 3 points for a mask.")
            return
            
        # default class 0 for gallstone
        class_id = 0 
        new_coords = []
        for x, y in self.current_mask_pts:
            new_coords.extend([x / self.w, y / self.h])
            
        self.polygons.append((class_id, new_coords))
        self._save_labels()
        
        self.current_mask_pts = []
        self.mode = "VIEW"
        print("New mask saved.")

    def run(self):
        if self.total_images == 0:
            print("No images found.")
            return

        while self.current_idx < self.total_images:
            self.img_path = self.image_paths[self.current_idx]
            self.label_path = Path(str(self.img_path).replace("images", "labels").rsplit(".", 1)[0] + ".txt")
            
            self.img = cv2.imread(str(self.img_path))
            if self.img is None:
                self.current_idx += 1
                continue
                
            self.h, self.w = self.img.shape[:2]
            self._load_labels()
            self.mode = "VIEW"
            
            while True:
                self._refresh_display()
                key = cv2.waitKey(10) & 0xFF
                
                # Global keys
                if key == 27 or key == ord('q'): # ESC or Q
                    print(f"\nReview stopped. Deleted {self.deleted_count} images.")
                    cv2.destroyAllWindows()
                    return
                    
                if self.mode == "VIEW":
                    if key == 32 or key == ord('n'): # Space or N
                        self.current_idx += 1
                        break
                    elif key == ord('b'): # B for Back
                        if self.current_idx > 0:
                            self.current_idx -= 1
                        break
                    elif key == 8 or key == ord('d'): # Backspace or D
                        print(f"Deleting: {self.img_path.name}")
                        if self.img_path.exists(): os.remove(self.img_path)
                        if self.label_path.exists(): os.remove(self.label_path)
                        
                        if self.delete_processed and self.processed_dir.exists():
                            split_name = self.img_path.parent.parent.name
                            p_img = self.processed_dir / split_name / "images" / self.img_path.name
                            p_lbl = self.processed_dir / split_name / "labels" / self.label_path.name
                            if p_img.exists(): os.remove(p_img)
                            if p_lbl.exists(): os.remove(p_lbl)
                            
                        self.deleted_count += 1
                        self.current_idx += 1
                        break
                    elif key == ord('c'):
                        self.mode = "CROP"
                    elif key == ord('m'):
                        self.mode = "MASK"
                        
                elif self.mode == "CROP":
                    if key == ord('v'):
                        self.mode = "VIEW"
                        self.crop_start = None
                        self.crop_end = None
                    elif key == ord('s'):
                        self._apply_crop()
                        
                elif self.mode == "MASK":
                    if key == ord('v'):
                        self.mode = "VIEW"
                        self.current_mask_pts = []
                    elif key == ord('s'):
                        self._apply_mask()

        print(f"\nReview complete! Deleted {self.deleted_count} images.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset_final")
    parser.add_argument("--delete_processed", action="store_true")
    args = parser.parse_args()
    
    cleaner = DatasetCleaner(args.data_dir, args.delete_processed)
    cleaner.run()
