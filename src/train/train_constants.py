import torch
from src import constants

BATCH_SIZE = 18 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 500  # Number of epochs to train for.
NUM_WORKERS = 6 # Number of parallel workers for data loading.
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Training images and XML files directory.
TRAIN = constants.TRAIN_IMAGES_PATH
# Validation images and XML files directory.
VALID = constants.VAL_IMAGES_PATH
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__',
    'RBC'
]
NUM_CLASSES = len(CLASSES)
# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False
# Location to save model and plots.
OUT_DIR = 'outputs'
