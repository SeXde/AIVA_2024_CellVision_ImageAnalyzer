import glob
import os.path
import unittest
import xml.etree.ElementTree as et
import cv2

import src.cell_detector as cell_detector
import test_constants
import src.utils.io_utils as io_utils
import src.utils.model_utils as model_utils


def draw_bounding_boxes(image, xml_path):
    tree = et.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


class TestCellDetector(unittest.TestCase):
    def setUp(self):
        jpg_files = sorted(glob.glob(f'{os.path.join(test_constants.IMAGES_PATH, "*.jpg")}'))
        xml_files = sorted(glob.glob(f'{os.path.join(test_constants.IMAGES_PATH, "*.xml")}'))

        self.images_and_n_cells = []

        for jpg_file, xml_file in zip(jpg_files, xml_files):
            tree = et.parse(xml_file)
            root = tree.getroot()
            n_cels = len(root.findall(".//object"))

            self.images_and_n_cells.append((jpg_file, n_cels, xml_file))

        self.cell_detector = cell_detector.CellDetector()

    def test_detect_cells(self):
        for i, (image_path, n_cels, xml_file) in enumerate(self.images_and_n_cells):
            with self.subTest(msg=f'Checking image {i}'):
                image = io_utils.load_image(image_path)
                bboxes = self.cell_detector.detect_cells(image)
                self.assertEqual(n_cels, len(bboxes))
                real_cels = image.copy()
                draw_bounding_boxes(real_cels, xml_file)
                cv2.imshow('GT', real_cels)
                model_utils.print_bboxes(image, bboxes)
