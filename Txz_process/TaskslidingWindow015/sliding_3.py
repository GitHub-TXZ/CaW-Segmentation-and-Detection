# -*- coding: utf-8 -*-
from __future__ import absolute_import
from Txz_process.utils import *
import pathlib
import shutil
import os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import OrderedDict
import numpy as np
import pickle
import SimpleITK as sitk


# 就地标准化文件名格式，并且生成json文件
def generate_json(dir_my_data_cta, dir_my_data_seg, dir_nnunet):
    # 生成nnunet训练和测试所需要的json文件
    p_mydata_cta = Path(dir_my_data_cta)
    p_mydata_seg = Path(dir_my_data_seg)
    join = os.path.join
    files_cta1 = sorted(p_mydata_cta.rglob("*.nii.gz"), key=lambda x: x.as_posix()[:-12])
    files_seg = sorted(p_mydata_seg.rglob("*.nii.gz"))
    json_dict = {}
    json_dict['name'] = "Tan Xian Zhen"
    json_dict['description'] = "slide web segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "TXZ"
    json_dict['licence'] = "TXZ"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CTA",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "web",
    }
    json_dict['numTraining'] = len(files_cta1)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': join("imagesTr/", i.parts[-1][:-12] + ".nii.gz"),
                              "label": join("labelsTr/", j.parts[-1])} for
                             i, j in
                             zip(files_cta1, files_seg)]
    json_dict['test'] = []
    save_json(json_dict, join(dir_nnunet, "dataset.json"))


dir_my_data_cta = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/imagesTr"
dir_my_data_seg = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/labelsTr"
dir_nnunet = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu"
generate_json(dir_my_data_cta, dir_my_data_seg, dir_nnunet)
