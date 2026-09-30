import albumentations as A
import cv2
import numpy as np


class AcousticShadowAugment(A.ImageOnlyTransform):
    """Vary dark vertical shadow columns while preserving morphology."""

    def __init__(self, shadow_intensity_range=(0.3, 0.8), p=0.5):
        super().__init__(p=p)
        self.shadow_intensity_range = shadow_intensity_range

    def apply(self, image, shadow_intensity=0.6, **params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        col_means = np.mean(gray, axis=0)
        shadow_cols = np.where(col_means < np.percentile(col_means, 20))[0]
        result = image.copy().astype(np.float32)
        result[:, shadow_cols] *= shadow_intensity
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_params(self):
        return {"shadow_intensity": float(np.random.uniform(*self.shadow_intensity_range))}
