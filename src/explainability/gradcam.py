class GradCAMExplainer:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers

    def explain(self, image, method="gradcam++"):
        from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        methods = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "eigencam": EigenCAM, "layercam": LayerCAM}
        with methods[method](model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=image)
            np_img = image[0].permute(1, 2, 0).detach().cpu().numpy()
            visualization = show_cam_on_image(np_img, grayscale_cam[0], use_rgb=True)
        return visualization, grayscale_cam
