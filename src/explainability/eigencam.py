from .gradcam import GradCAMExplainer


class EigenCAMExplainer(GradCAMExplainer):
    def explain(self, image):
        return super().explain(image, method="eigencam")
