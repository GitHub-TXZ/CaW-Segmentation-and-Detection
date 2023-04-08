import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()




def get_label(dir):
    have_web = []
    files = glob.glob(os.path.join(dir, "*.nii.gz"))
    for file in files:
        if 1 in sitk.GetArrayFromImage(sitk.ReadImage(file)):
            have_web.append(file)
    print(len(have_web))
    return have_web

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.data = []
        have_web = get_label("/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task999_bifurcation_proc/labelsTr")
        for file in glob.glob(os.path.join(data_dir, "*.nii.gz")):
            label = 1 if file.replace("imagesTr", "labelsTr").replace("_0000",'') in have_web else 0
            self.data.append((file, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_file, label = self.data[index]
        image = LoadImage(image_file)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

data_dir = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task999_bifurcation_proc/imagesTr"
myDataset = MyDataset(data_dir)
print(myDataset.data)
myDataset.__getitem__(0)