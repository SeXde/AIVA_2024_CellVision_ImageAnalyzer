import glob
import os.path
import unittest
import xml.etree.ElementTree as et

import src.cell_detector as cell_detector
import test_constants
import src.utils.io_utils as io_utils


class TestCellDetector(unittest.TestCase):
    def setUp(self):
        jpg_files = sorted(glob.glob(f'{os.path.join(test_constants.IMAGES_PATH, "*.jpg")}'))
        xml_files = sorted(glob.glob(f'{os.path.join(test_constants.IMAGES_PATH, "*.xml")}'))

        self.images_and_n_cells = []

        for jpg_file, xml_file in zip(jpg_files, xml_files):

            tree = et.parse(xml_file)
            root = tree.getroot()
            n_cels = len(root.findall(".//object"))

            self.images_and_n_cells.append((jpg_file, n_cels))

        self.cell_detector = cell_detector.CellDetector()

    def test_detect_cells(self):
        for i, (image_path, n_cels) in enumerate(self.images_and_n_cells):
            with self.subTest(msg=f'Checking image {i}'):
                self.assertEqual(n_cels, len(self.cell_detector.detect_cells(io_utils.load_image(image_path))))
