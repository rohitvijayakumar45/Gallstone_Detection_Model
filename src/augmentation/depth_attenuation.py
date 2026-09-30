import albumentations as A
import numpy as np


class DepthAttenuationTransform(A.ImageOnlyTransform):
    """Simulate ultrasound signal attenuation with depth."""

    def __init__(self, max_attenuation=0.4, p=0.5):
        super().__init__(p=p)
        self.max_attenuation = max_attenuation

    def apply(self, image, **params):
        h, w = image.shape[:2]
        gradient = np.linspace(1.0, 1.0 - self.max_attenuation, h)
        mask = np.tile(gradient.reshape(-1, 1), (1, w))
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
        return np.clip(image.astype(np.float32) * mask, 0, 255).astype(image.dtype)
