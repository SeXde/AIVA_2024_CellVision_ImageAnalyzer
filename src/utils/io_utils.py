import numpy as np
import cv2


def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path, cv2.IMREAD_COLOR)
