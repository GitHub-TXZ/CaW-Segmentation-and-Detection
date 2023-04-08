# -*-coding:utf-8 -*-
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import os
import shutil
from pathlib import Path
import sys
import SimpleITK as sitk
from Txz_process.utils import save_json
from Txz_process.utils import set_meta

dataset_dir = Path("/homeb/txz/Pycharm_Project/Txz_process/raw_data")
images = sorted(dataset_dir.rglob('*cta1.nii.gz'))
segs = sorted(dataset_dir.rglob('*seg.nii.gz'))
for file_seg, file_im in zip(segs, images):
    shutil.copy(file_seg, "/homeb/txz/Pycharm_Project/Txz_process/Task_ultimate_procedure/raw_seg/" + file_seg.parts[
        -2] + ".nii.gz")
    shutil.copy(file_im, "/homeb/txz/Pycharm_Project/Txz_process/Task_ultimate_procedure/raw_cta/" + file_im.parts[
        -2] + ".nii.gz")
    print("Next file")
