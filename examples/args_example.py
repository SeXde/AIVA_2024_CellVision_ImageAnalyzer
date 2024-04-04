import sys

from src.cell_detector import CellDetector
from src.utils import io_utils, model_utils


"""
Basic example on how to use the CellDetector class with an image path given from the run arguments. 
"""

# Initialize the cell detector
detector = CellDetector()

# Load the image as a np.ndarray
image_path = sys.argv[1]  # Read the first argument
if not image_path:
    sys.exit("Missing image_path argument. Usage: `python args_example.py path_to_image`")

image = io_utils.load_image(image_path)

# Run the detection and return the bounding boxes
bounding_boxes = detector.detect_cells(image)

# You can draw the bounding boxes using our function or your own
# (you can use our as a base)
model_utils.print_bboxes(image, bounding_boxes)
