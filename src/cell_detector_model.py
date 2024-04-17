from abc import ABC, abstractmethod
import numpy as np
import utils.model_utils as model_utils
import train.train_constants as train_constants
import torch
import constants
import cv2


class CellDetectorModel(ABC):
    def __init__(self):
        self.load_trained_model()

    @abstractmethod
    def load_trained_model(self):
        pass

    @abstractmethod
    def get_detections(self, image):
        pass


class FCOSCellDetectorModel(CellDetectorModel):

    def __init__(self):
        self.model = None
        super().__init__()

    def load_trained_model(self):
        self.model = model_utils.create_fcos_model(num_classes=train_constants.NUM_CLASSES,
                                                   min_size=train_constants.RESIZE_TO,
                                                   max_size=train_constants.RESIZE_TO).to(train_constants.DEVICE)
        checkpoint = torch.load(constants.MODEL_FILE_PATH, map_location=train_constants.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_detections(self, image: np.ndarray) -> list:
        orig_image = image.copy()
        image = cv2.resize(image, (train_constants.RESIZE_TO, train_constants.RESIZE_TO))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image_input = torch.tensor(image_input, dtype=torch.float).to(train_constants.DEVICE)
        image_input = torch.unsqueeze(image_input, 0)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(image_input.to(train_constants.DEVICE))
        return model_utils.model_outputs_to_bbox(outputs, orig_image, image)
