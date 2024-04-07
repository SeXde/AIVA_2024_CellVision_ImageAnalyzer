import os
from pathlib import Path


# Paths of project
SRC_PATH = Path(__file__).parent
RESOURCES_PATH = os.path.join(SRC_PATH, 'resources')
TRAIN_PATH = os.path.join(SRC_PATH, 'train')
IMAGES_PATH = os.path.join(RESOURCES_PATH, 'images')
TRAIN_IMAGES_PATH = os.path.join(IMAGES_PATH, 'train')
VAL_IMAGES_PATH = os.path.join(IMAGES_PATH, 'val')
OUTPUT_FILE_PATH = os.path.join(TRAIN_PATH, 'outputs')
MODEL_FILE_PATH = os.path.join(OUTPUT_FILE_PATH, 'last_model.pth')
