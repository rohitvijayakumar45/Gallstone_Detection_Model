import cv2


class SpeckleReducer:
    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, gray):
        return cv2.bilateralFilter(gray, self.d, self.sigma_color, self.sigma_space)
