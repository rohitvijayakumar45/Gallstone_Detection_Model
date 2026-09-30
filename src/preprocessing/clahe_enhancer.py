import cv2


class CLAHEEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def __call__(self, gray):
        return self.clahe.apply(gray)
