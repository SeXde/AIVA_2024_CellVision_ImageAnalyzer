import unittest
import src.constants as constants
import os
import test_constants
import src.train.train_constants as train_constants
import src.train.datasets as datasets


class TestDataset(unittest.TestCase):

    @staticmethod
    def count_files(dir_path):
        file_count_dict = {}
        for file_name in os.listdir(dir_path):
            file_name = file_name.replace('.jpg', '').replace('.xml', '')
            if file_count_dict.get(file_name) is not None:
                file_count_dict[file_name] += 1
            else:
                file_count_dict[file_name] = 1
        return file_count_dict

    def test_dataset_integrity(self):
        file_count_dict_train = self.count_files(constants.TRAIN_IMAGES_PATH)
        for count in file_count_dict_train.values():
            self.assertEqual(2, count)

        file_count_dict_val = self.count_files(constants.VAL_IMAGES_PATH)
        for count in file_count_dict_val.values():
            self.assertEqual(2, count)

        file_count_dict_test = self.count_files(test_constants.IMAGES_PATH)
        for count in file_count_dict_test.values():
            self.assertEqual(2, count)

    def test_dataset_creation(self):
        dataset_train = datasets.create_train_dataset(constants.TRAIN_IMAGES_PATH, train_constants.CLASSES,
                                                      train_constants.RESIZE_TO)
        loader_train = datasets.create_train_loader(dataset_train, batch_size=2, num_workers=0)

        train_images_len = len(os.listdir(constants.TRAIN_IMAGES_PATH))
        self.assertEqual(train_images_len, len(loader_train))

        dataset_valid = datasets.create_valid_dataset(constants.VAL_IMAGES_PATH, train_constants.CLASSES,
                                                      train_constants.RESIZE_TO)
        loader_valid = datasets.create_valid_loader(dataset_valid, batch_size=2, num_workers=0)

        val_images_len = len(os.listdir(constants.VAL_IMAGES_PATH))
        self.assertEqual(val_images_len, len(loader_valid))
