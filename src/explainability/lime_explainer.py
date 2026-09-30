from lime import lime_image
from skimage.segmentation import mark_boundaries, slic


class LIMEExplainer:
    def __init__(self, predict_fn, num_superpixels=50):
        self.explainer = lime_image.LimeImageExplainer()
        self.predict_fn = predict_fn
        self.num_superpixels = num_superpixels

    def explain(self, image_np, num_samples=200, positive_only=True):
        explanation = self.explainer.explain_instance(
            image_np,
            self.predict_fn,
            top_labels=1,
            num_samples=num_samples,
            num_features=self.num_superpixels,
            segmentation_fn=lambda x: slic(x, n_segments=self.num_superpixels, compactness=30),
        )
        temp_image, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=positive_only,
            num_features=10,
            hide_rest=False,
        )
        return mark_boundaries(temp_image / 255.0, mask), explanation
