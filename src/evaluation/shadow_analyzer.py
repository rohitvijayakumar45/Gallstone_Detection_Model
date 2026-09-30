import cv2
import numpy as np

class ShadowAnalyzer:
    """Analyzes acoustic shadowing below gallstone detections."""
    
    def __init__(self, shadow_depth=100, threshold=0.7):
        self.shadow_depth = shadow_depth
        self.threshold = threshold # Ratio of shadow intensity to local background

    def verify_detection(self, image, box):
        """
        Returns a confidence multiplier based on shadow presence.
        box: [x1, y1, x2, y2] in pixels.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        h, w = gray.shape
        x1, y1, x2, y2 = map(int, box)
        
        # 1. Define shadow region (directly below the stone)
        sy1 = y2
        sy2 = min(h, y2 + self.shadow_depth)
        sx1, sx2 = x1, x2
        
        if sy2 <= sy1 or sx2 <= sx1:
            return 1.0 # Can't verify
            
        shadow_roi = gray[sy1:sy2, sx1:sx2]
        shadow_mean = np.mean(shadow_roi)
        
        # 2. Define background regions (to the left and right of the shadow)
        # We look for the 'unshadowed' intensity at the same depth
        margin = (x2 - x1) // 2
        bg_left = gray[sy1:sy2, max(0, x1 - margin):x1]
        bg_right = gray[sy1:sy2, x2:min(w, x2 + margin)]
        
        bg_means = []
        if bg_left.size > 0: bg_means.append(np.mean(bg_left))
        if bg_right.size > 0: bg_means.append(np.mean(bg_right))
        
        if not bg_means:
            return 1.0
            
        bg_mean = np.mean(bg_means)
        
        # 3. Calculate shadow ratio
        # A true shadow should be significantly darker than the surroundings
        ratio = shadow_mean / (bg_mean + 1e-6)
        
        if ratio < self.threshold:
            # Strong shadow detected! Boost confidence.
            return 1.2
        elif ratio > 0.95:
            # No shadow at all. Likely a false positive (e.g. noise or bowel gas).
            return 0.5
        else:
            return 1.0

    def apply_to_ensemble(self, image, boxes, scores):
        """
        Adjusts ensemble scores based on shadow verification.
        boxes: absolute pixels [x1, y1, x2, y2]
        """
        new_scores = []
        for box, score in zip(boxes, scores):
            multiplier = self.verify_detection(image, box)
            new_scores.append(min(1.0, score * multiplier))
        return np.array(new_scores)
