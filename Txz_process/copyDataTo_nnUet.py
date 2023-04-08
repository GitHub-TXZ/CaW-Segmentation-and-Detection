# -*- coding: utf-8 -*-
from __future__ import absolute_import
from utils import *
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


def copy_data_to_nnUnet(dir_my_data, dir_nnunet, do_split, kflod):
    """
    eg:# base = "/homeb/txz/pytorch-UNet/data_processed2"  # 原始文件路径
       # out = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid"   # 结构化数据集目录
    :param dir_my_data:
    :param dir_nnunet:
    :return:
    """
    global Ts_file_cta1, Tr_file_seg
    join = os.path.join
    maybe_mkdir_p(dir_nnunet)
    maybe_mkdir_p(join(dir_nnunet, "imagesTr"))
    maybe_mkdir_p(join(dir_nnunet, "labelsTr"))
    maybe_mkdir_p(join(dir_nnunet, "imagesTs"))
    maybe_mkdir_p(join(dir_nnunet, "labelsTs"))
    maybe_mkdir_p(join(dir_nnunet, "prediction"))

    p_mydata = Path(dir_my_data)
    files_cta1 = sorted(p_mydata.rglob("*cta1.nii.gz"))
    num_total = len(files_cta1)
    Tr_Ts_info = {}
    files_seg = sorted(p_mydata.rglob("*seg.nii.gz"))
    if kflod == 0:
        Tr_file_cta1 = files_cta1[0:num_total // do_split * 2]
        Tr_file_seg = files_seg[0:num_total // do_split * 2]
        Ts_file_cta1 = files_cta1[num_total // do_split * 2:]
        Ts_file_seg = files_seg[num_total // do_split * 2:]
        Tr_Ts_info["kflod"] = kflod
        Tr_Ts_info["Tr_file_cta1"] = str(Tr_file_cta1)
        Tr_Ts_info["Tr_file_seg"] = str(Tr_file_seg)
        Tr_Ts_info["Ts_file_cta1"] = str(Ts_file_cta1)
        Tr_Ts_info["Ts_file_seg"] = str(Ts_file_seg)
    elif kflod == 1:
        Ts_file_cta1 = files_cta1[num_total // do_split * 1:num_total // do_split * 2]
        Ts_file_seg = files_seg[num_total // do_split * 1:num_total // do_split * 2]
        tmp = files_cta1[num_total // do_split * 2:]
        tmp.extend(files_cta1[0:num_total // do_split * 1])
        Tr_file_cta1 = tmp
        tmp = files_seg[num_total // do_split * 2:]
        tmp.extend(files_seg[0:num_total // do_split * 1])
        Tr_file_seg = tmp
        Tr_Ts_info["kflod"] = kflod
        Tr_Ts_info["Tr_file_cta1"] = str(Tr_file_cta1)
        Tr_Ts_info["Tr_file_seg"] = str(Tr_file_seg)
        Tr_Ts_info["Ts_file_cta1"] = str(Ts_file_cta1)
        Tr_Ts_info["Ts_file_seg"] = str(Ts_file_seg)

    elif kflod == 2:
        Tr_file_cta1 = files_cta1[num_total // do_split * 1:]
        Tr_file_seg = files_seg[num_total // do_split * 1:]
        Ts_file_cta1 = files_cta1[0:num_total // do_split * 1]
        Ts_file_seg = files_seg[0:num_total // do_split * 1]
        Tr_Ts_info["kflod"] = kflod
        Tr_Ts_info["Tr_file_cta1"] = str(Tr_file_cta1)
        Tr_Ts_info["Tr_file_seg"] = str(Tr_file_seg)
        Tr_Ts_info["Ts_file_cta1"] = str(Ts_file_cta1)
        Tr_Ts_info["Ts_file_seg"] = str(Ts_file_seg)
    save_json(Tr_Ts_info, f"{dir_my_data}_kflod = {kflod}_data_split.json")
    for file in Tr_file_cta1:
        shutil.copy(file, join(dir_nnunet, "imagesTr", file.stem.split('.')[0].rstrip("_cta1") + "_0000.nii.gz"))
    for file in Tr_file_seg:
        shutil.copy(file, join(dir_nnunet, "labelsTr", file.stem.split('.')[0].rstrip("_seg") + ".nii.gz"))
    for file in Ts_file_cta1:
        shutil.copy(file, join(dir_nnunet, "imagesTs", file.stem.split('.')[0].rstrip("_cta1") + "_0000.nii.gz"))
    for file in Ts_file_seg:
        shutil.copy(file, join(dir_nnunet, "labelsTs", file.stem.split('.')[0].rstrip("_seg") + ".nii.gz"))
    # 生成nnunet训练和测试所需要的json文件
    json_dict = {}
    json_dict['name'] = "cws"
    json_dict['description'] = "carotid lumen segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "cws data for nnunet"
    json_dict['licence'] = "dog_shit"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CTA",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "carotid lumen",
        #     "2": "carotid lumen left",
        #     "3": "web right",
        #     "4": "web left"
    }
    json_dict['numTraining'] = len(Tr_file_cta1)
    json_dict['numTest'] = len(Ts_file_cta1)
    json_dict['training'] = [{'image': join("imagesTr/", i.stem.split(".")[0].rstrip("_cta1") + ".nii.gz"),
                              "label": join("labelsTr/", j.stem.split(".")[0].rstrip("_seg") + ".nii.gz")} for i, j in
                             zip(Tr_file_cta1, Tr_file_seg)]
    json_dict['test'] = [join("imagesTs/", i.stem.split(".")[0].rstrip("_cta1") + ".nii.gz") for i in Ts_file_cta1]

    save_json(json_dict, join(dir_nnunet, "dataset.json"))


# 重新写的
class my_cta_seg_task:
    def __init__(self, dir_my_data, dir_nnunet):
        self.dir_my_data = dir_my_data
        self.dir_nnunet = dir_nnunet

    def new_copy_data_to_nnUnet(self):
        join = os.path.join
        maybe_mkdir_p(self.dir_nnunet)
        maybe_mkdir_p(join(self.dir_nnunet, "imagesTr"))
        maybe_mkdir_p(join(self.dir_nnunet, "labelsTr"))
        maybe_mkdir_p(join(self.dir_nnunet, "imagesTs"))
        maybe_mkdir_p(join(self.dir_nnunet, "labelsTs"))
        maybe_mkdir_p(join(self.dir_nnunet, "prediction"))

        p_mydata = Path(self.dir_my_data)
        files_cta1 = sorted(p_mydata.rglob("*cta1.nii.gz"))
        files_seg = sorted(p_mydata.rglob("*seg.nii.gz"))
        for file_cta in files_cta1:
            shutil.copy(file_cta, join(self.dir_nnunet, "imagesTr", file_cta.parts[-1][:-7] + "_0000.nii.gz"))
        for file_seg in files_seg:
            shutil.copy(file_seg, join(self.dir_nnunet, "labelsTr", file_seg.parts[-1][:-7] + ".nii.gz"))

    def generate_json(self):
        # 生成nnunet训练和测试所需要的json文件
        p_mydata = Path(self.dir_my_data)
        join = os.path.join
        files_cta1 = sorted(p_mydata.rglob("*cta1.nii.gz"))
        files_seg = sorted(p_mydata.rglob("*seg.nii.gz"))
        json_dict = {}
        json_dict['name'] = "Tan Xian Zhen"
        json_dict['description'] = "carotid lumen and web segmentation"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "TXZ"
        json_dict['licence'] = "TXZ"
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "CTA",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "carotid lumen",
        }
        json_dict['numTraining'] = len(files_cta1)
        json_dict['numTest'] = 0
        json_dict['training'] = [{'image': join("imagesTr/", i.parts[-1][:-12] + ".nii.gz"),
                                  "label": join("labelsTr/", j.parts[-1][:-11] + ".nii.gz")} for
                                 i, j in
                                 zip(files_cta1, files_seg)]
        json_dict['test'] = []
        save_json(json_dict, join(dir_nnunet, "dataset.json"))

    def do_my_split(self, k: int = 5):
        path = self.dir_my_data
        p = Path(path)
        files = p.rglob("*cta1.nii.gz")
        li = []
        for i in files:
            li.append(i.parts[-1][:-12])
        li.sort()
        kFold = KFold(n_splits=k, shuffle=False)
        splits = []
        pat_ids = np.asarray(li)
        for train_idx, test_idx in kFold.split(pat_ids):
            train_idx, val_idx = train_test_split(train_idx, test_size=0.2, shuffle=False)
            d = OrderedDict()
            d['train'] = pat_ids[train_idx]
            d['val'] = pat_ids[val_idx]
            d['test'] = pat_ids[test_idx]
            splits.append(d)
        with open(os.path.join(self.dir_nnunet[0:27], 'nnUNet_preprocessed', dir_nnunet.split('/')[-1],
                               'splits_final.pkl'),
                  'wb') as f:
            pickle.dump(splits, f)


class my_cta_seg_task_new:
    def __init__(self, dir_my_data_cta, dir_my_data_seg, dir_nnunet):
        self.dir_my_data_cta = dir_my_data_cta
        self.dir_my_data_seg = dir_my_data_seg
        self.dir_nnunet = dir_nnunet

    def new_copy_data_to_nnUnet(self):
        join = os.path.join
        maybe_mkdir_p(self.dir_nnunet)
        maybe_mkdir_p(join(self.dir_nnunet, "imagesTr"))
        maybe_mkdir_p(join(self.dir_nnunet, "labelsTr"))
        maybe_mkdir_p(join(self.dir_nnunet, "imagesTs"))
        maybe_mkdir_p(join(self.dir_nnunet, "labelsTs"))
        maybe_mkdir_p(join(self.dir_nnunet, "prediction"))

        p_mydata_cta = Path(self.dir_my_data_cta)
        p_mydata_seg = Path(self.dir_my_data_seg)
        files_cta1 = sorted(p_mydata_cta.rglob("*.nii.gz"))
        files_seg = sorted(p_mydata_seg.rglob("*.nii.gz"))
        for file_cta in files_cta1:
            shutil.copy(file_cta, join(self.dir_nnunet, "imagesTr", file_cta.parts[-1][:-7] + "_0000.nii.gz"))
        for file_seg in files_seg:
            shutil.copy(file_seg, join(self.dir_nnunet, "labelsTr", file_seg.parts[-1][:-7] + ".nii.gz"))

    def generate_json(self):
        # 生成nnunet训练和测试所需要的json文件
        p_mydata_cta = Path(self.dir_my_data_cta)
        p_mydata_seg = Path(self.dir_my_data_seg)
        join = os.path.join
        files_cta1 = sorted(p_mydata_cta.rglob("*.nii.gz"))
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
        json_dict['training'] = [{'image': join("imagesTr/", i.parts[-1][:-7] + ".nii.gz"),
                                  "label": join("labelsTr/", j.parts[-1][:-7] + ".nii.gz")} for
                                 i, j in
                                 zip(files_cta1, files_seg)]
        json_dict['test'] = []
        save_json(json_dict, join(dir_nnunet, "dataset.json"))


if __name__ == '__main__':
    # dir_my_data = "./data_patch_web_small"
    # dir_nnunet = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task014_carotid"
    # my_cta_seg_task = my_cta_seg_task(dir_my_data, dir_nnunet)
    # my_cta_seg_task.new_copy_data_to_nnUnet()
    # my_cta_seg_task.generate_json()
    # import os
    # import sys
    # os.system("ls")
    # os.system("conda activate cta-seg")
    # os.system('/homeb/txz/anaconda3/envs/cta-seg/bin/python3 /homeb/txz/Pycharm_Project/nnUNet/nnunet/experiment_planning/nnUNet_plan_and_preprocess.py  -t 14 --verify_dataset_integrity')
    # my_cta_seg_task.do_my_split(k=3)

    # dir_nnunet = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task%03d_carotid" % kaishide_TaskID
    # copy_data_to_nnUnet(dir_my_data, dir_nnunet, 3, 0) # 分成几份，取哪一份
    # dir_nnunet = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task%03d_carotid" % (kaishide_TaskID + 1)
    # copy_data_to_nnUnet(dir_my_data, dir_nnunet, 3, 1)  # 分成几份，取哪一份
    # dir_nnunet = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task%03d_carotid" % (kaishide_TaskID + 2)
    # copy_data_to_nnUnet(dir_my_data, dir_nnunet, 3, 2)  # 分成几份，取哪一份
    dir_my_data_cta = "/homeb/txz/Pycharm_Project/Txz_process/Task_ultimate_procedure/neck_cta"
    dir_my_data_seg = "/homeb/txz/Pycharm_Project/Txz_process/Task_ultimate_procedure/neck_seg"
    dir_nnunet = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task666_utimate_proc"
    my_cta_seg_task_new = my_cta_seg_task_new(dir_my_data_cta, dir_my_data_seg, dir_nnunet)
    my_cta_seg_task_new.new_copy_data_to_nnUnet()
    my_cta_seg_task_new.generate_json()
