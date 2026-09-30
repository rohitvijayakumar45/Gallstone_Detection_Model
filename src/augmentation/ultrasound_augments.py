import albumentations as A
import numpy as np


class SpeckleNoiseTransform(A.ImageOnlyTransform):
    """Multiplicative ultrasound speckle noise: J = I + n * I."""

    def __init__(self, mean=0.0, std_range=(0.01, 0.05), p=0.5):
        super().__init__(p=p)
        self.mean = mean
        self.std_range = std_range

    def apply(self, image, noise_std=0.02, **params):
        noise = np.random.normal(self.mean, noise_std, image.shape)
        speckle = image.astype(np.float32) + image.astype(np.float32) * noise
        return np.clip(speckle, 0, 255).astype(np.uint8)

    def get_params(self):
        return {"noise_std": float(np.random.uniform(*self.std_range))}


class ReverberationArtifact(A.ImageOnlyTransform):
    """Horizontal bright line artifacts common in abdominal ultrasound."""

    def apply(self, image, **params):
        result = image.copy().astype(np.float32)
        h = image.shape[0]
        n_lines = np.random.randint(1, 4)
        for _ in range(n_lines):
            y = np.random.randint(0, max(1, h // 3))
            thickness = np.random.randint(1, 3)
            intensity = np.random.randint(30, 90)
            result[y : y + thickness, :] += intensity
        return np.clip(result, 0, 255).astype(np.uint8)
