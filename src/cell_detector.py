import numpy as np
import cell_detector_model


class CellDetector:
    def __init__(self):
        self.model = cell_detector_model.FCOSCellDetectorModel()

    def detect_cells(self, image: np.ndarray) -> list:
        return self.model.get_detections(image)
