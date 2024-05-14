import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import ast
import torchvision.transforms as transforms
import pandas as pd
from torchvision.io import read_image

class LabelMeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, start_image_idx, transform=None, target_transform=None):
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.start = start_image_idx
        self.img_names, self.img_labels = self.get_labels(annotations_file)

    def get_labels(self, annotations_file):
        class_labels = np.fromfile(annotations_file, dtype=np.float32).reshape(-1, 12)
        correct_labels = np.argmax(class_labels, axis = 1)
        # print(class_labels, correct_labels)
        required_width = 6
        img_names = [str(i).zfill(required_width) for i in range(self.start, self.start + len(correct_labels))]
        return img_names, correct_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        val = idx // 1000
        subfolder = str(val).zfill(4)
        sub_dir = self.img_dir + "/" + subfolder
        img_path = os.path.join(sub_dir, self.img_names[idx] + ".jpg")
        # print(img_path)
        # image = read_image(img_path)
        image = Image.open(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

