import os
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as et
import src.train.train_utils as train_utils
import src.utils.io_utils as io_utils


class CustomDataset(Dataset):
    def __init__(self, img_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.img_path = img_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []

        # Get all image paths in sorted order
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.img_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.img_path, image_name)
        if not os.path.exists(image_path):
            raise ValueError(f'Image "{image_path}" not found')
        image = io_utils.load_image(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.img_path, annot_filename)
        if not os.path.exists(annot_file_path):
            raise ValueError(f'Annotation "{annot_file_path}" not found')
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            if xmax_final == xmin_final:
                xmin_final -= 1
            if ymax_final == ymin_final:
                ymin_final -= 1
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=image_resized, bboxes=target['boxes'], labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)


def create_train_dataset(img_dir, classes, resize=640):
    train_dataset = CustomDataset(img_dir, resize, resize, classes, train_utils.get_train_transform())
    return train_dataset


def create_valid_dataset(img_dir, classes, resize=640):
    valid_dataset = CustomDataset(img_dir, resize, resize, classes, train_utils.get_valid_transform())
    return valid_dataset


def create_train_loader(train_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=train_utils.collate_fn, drop_last=False)
    return train_loader


def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=train_utils.collate_fn, drop_last=False)
    return valid_loader
