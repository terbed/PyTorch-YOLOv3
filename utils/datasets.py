import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import h5py
import cv2

from utils.augmentations import horizontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms
tr = torch


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size, mode='nearest'):
    image = F.interpolate(image.unsqueeze(0), size=size, mode=mode).squeeze(0)
    return image


def file_is_empty(filename):
    with open(filename) as fin:
        for line in fin:
            line = line[:line.find('#')]  # remove '#' comments
            line = line.strip()  # rmv leading/trailing white space
            if len(line) != 0:
                return False
    return True


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416, resize_mode='nearest'):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        self.resize_mode = resize_mode

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size, self.resize_mode)
        # channel centralization
        # img = torch.sub(img, torch.mean(img, (1, 2)).view(3, 1, 1))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, resize_mode='nearest', augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.resize_mode = resize_mode
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # Open image
        img = Image.open(img_path).convert('RGB')
        # Apply PIL augmentations
        if self.augment:
            if np.random.random() < 0.8:
                img = transforms.ColorJitter(
                    brightness=(0.3, 1.5),
                    contrast=(0.7, 1.3),
                    saturation=(0.7, 1.3),
                    hue=(-0.1, 0.1)
                )(img)
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if not file_is_empty(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations on image tensor (and also on label correspondingly)
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horizontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images
        imgs = torch.stack([resize(img, self.img_size, self.resize_mode) for img in imgs])

        # Resize labels
        self.batch_count += 1

        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ListDatasetHDF5(Dataset):
    def __init__(self, path, img_size=416, resize_mode='nearest', normalized_labels=True, add_bboxes=True):
        self.add_bbox = add_bboxes
        self.path = path
        self.D = 128

        self.N = None
        if add_bboxes:
            # Create dataset for bounding boxes
            with h5py.File(path, 'a') as db:
                frames = db['frames']
                self.N = frames.shape[0]
                keys = db.keys()
                print(keys)
                is_there = 'bbox' in keys
                if is_there:
                    print('Dataset already created...')
                else:
                    db.create_dataset('bbox', shape=(self.N, 4), maxshape=(None, 4), dtype=np.int, chunks=True)
                    print('\nAdded empty bounding box dataset to hdf5!')

        self.img_size = img_size
        self.resize_mode = resize_mode
        self.max_objects = 5
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

        self.num_samples = ((self.N - 64) // self.D) - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # ---------
        #  Image
        # ---------
        img = None
        with h5py.File(self.path, 'r') as db:
            frames = db['frames']
            img = frames[index*self.D, :, :, :]

        # convert to 8 bit if needed
        if img.dtype is np.dtype(np.uint16):
            if np.max(img[:]) < 256:
                scale = 255.  # 8 bit stored as 16 bit...
            elif np.max(img[:]) < 4096:
                scale = 4095.  # 12 bit
            else:
                scale = 65535.  # 16 bit
            img = cv2.convertScaleAbs(img, alpha=(225. / scale))

        img = transforms.ToTensor()(img)
        img = F.interpolate(img.unsqueeze(0), size=416, mode="nearest").squeeze()
        img = img.type(tr.FloatTensor)

        return img

