from src.cell_detector import CellDetector
from src.utils import io_utils, model_utils

"""
Basic example on how to use the CellDetector class. 
"""

# Initialize the cell detector
detector = CellDetector()

# Load the image as a np.ndarray
# You can use our function or your own
image = io_utils.load_image("your_image_path")

# Run the detection and return the bounding boxes
bounding_boxes = detector.detect_cells(image)

# You can draw the bounding boxes using our function or your own
# (you can use our as a base)
model_utils.print_bboxes(image, bounding_boxes)
