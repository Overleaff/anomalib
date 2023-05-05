import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import glob
import imgaug.augmenters as iaa
# from perlin import rand_perlin_2d_np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pdb
import os
from PIL import Image


class ImageNetDataset(Dataset):
    def __init__(self, imagenet_dir,transform=None,):
        super().__init__()
        self.imagenet_dir = imagenet_dir
        self.transform = transform
        self.dataset = ImageFolder(self.imagenet_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]