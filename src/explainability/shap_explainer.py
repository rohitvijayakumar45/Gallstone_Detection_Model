import numpy as np
import torch


class SHAPExplainer:
    def __init__(self, model, background_images):
        import shap

        self.model = model
        self.explainer = shap.DeepExplainer(model, torch.stack(background_images))

    def explain(self, image):
        shap_values = self.explainer.shap_values(image.unsqueeze(0))
        shap_img = np.abs(shap_values[0]).mean(axis=0)
        return (shap_img - shap_img.min()) / max(shap_img.max() - shap_img.min(), 1e-8)
