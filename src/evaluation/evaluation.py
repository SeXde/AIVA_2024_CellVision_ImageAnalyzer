import glob
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import tensor
from torchmetrics.detection import MeanAveragePrecision

from src import constants
from src.cell_detector import CellDetector
from src.utils import io_utils

# Constants
LABEL = 'cell'


def parse_xml_annotations(xml_file):
    """
    Parse XML annotations file and extract bounding box annotations.

    Args:
        xml_file (str): Path to the XML annotations file.

    Returns:
        list: List of dictionaries containing bounding box annotations.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    labels = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        annotations.append([xmin, ymin, xmax, ymax])
        labels.append(0)
    return [
        dict(
            boxes=tensor(annotations),
            labels=tensor(labels),
        )
    ]


def get_map_eval(desired_detector, dataset_path, show_plot=True):
    """
    Calculate Mean Average Precision (mAP) for a given detector on a dataset.

    Args:
        desired_detector: Instance of the detector model.
        dataset_path (str): Path to the dataset directory.
        show_plot (bool): Whether to show the mAP plot. Default is True.

    Returns:
        float: Mean Average Precision (mAP) across the dataset.
    """
    test_images = sorted(glob.glob(dataset_path + '/*.jpg'))
    test_annotations = sorted(glob.glob(dataset_path + '/*.xml'))

    mAPs = []
    metric = MeanAveragePrecision(iou_type="bbox")

    for image_path, annotation_path in zip(test_images, test_annotations):
        image = io_utils.load_image(image_path)
        target = (parse_xml_annotations(annotation_path))
        predictions = desired_detector.detect_cells(image)
        annotations = []
        labels = []
        scores = []
        for (x_min, y_min), (x_max, y_max) in predictions:
            annotations.append([x_min, y_min, x_max, y_max])
            labels.append(0)
            scores.append(0.8)
        preds = [
            dict(
                boxes=tensor(annotations),
                labels=tensor(labels),
                scores=tensor(scores)
            )
        ]
        metric.update(preds, target)
        map = float(metric.compute()['map'].to('cpu'))
        mAPs.append(map)

    map_mean = np.mean(mAPs)

    if show_plot:
        sns.set_style("whitegrid")

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=range(len(mAPs)), y=mAPs, color='skyblue')
        plt.axhline(y=map_mean, color='r', linestyle='-', label=f'Mean mAP: {map_mean:.2f}')
        plt.xlabel('Image')
        plt.ylabel('mAP')
        plt.title('mAP per Image')
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plt.savefig('mAP_plot.png')

        # Show the plot
        plt.show()

    return map_mean


if __name__ == '__main__':
    # Initialize the detector
    detector = CellDetector()
    # Evaluate the detector on validation images
    get_map_eval(detector, constants.VAL_IMAGES_PATH)
