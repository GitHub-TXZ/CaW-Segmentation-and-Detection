# -*- coding: utf-8 -*-

import numpy as np
import os
import SimpleITK as sitk
from medpy import metric
from utils import save_json
from pathlib import Path
import collections
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def get_test_metrics(gt_label_path, pred_label_path):
    i1 = sitk.ReadImage(gt_label_path)
    gt_arr = sitk.GetArrayFromImage(i1)
    print("gt_arr.shape:", gt_arr.shape)
    i2 = sitk.ReadImage(pred_label_path)
    pred_arr = sitk.GetArrayFromImage(i2)
    print("pred_arr.shape:", pred_arr.shape)
    # pred_arr = pred_arr.astype(np.uint8)
    # gt_arr = gt_arr.astype(np.uint8)
    dice = metric.binary.dc(pred_arr, gt_arr)
    hd = metric.binary.hd95(pred_arr, gt_arr)
    jc = metric.binary.jc(pred_arr, gt_arr)
    asd = metric.binary.asd(pred_arr, gt_arr)
    vol_gt = np.sum(gt_arr == 1) * np.prod(i1.GetSpacing()) / 1000
    vol_pred = np.sum(pred_arr == 1) * np.prod(i2.GetSpacing()) / 1000
    return {"dice": dice, "hd": hd, "jc": jc, "asd": asd, "vol_gt": vol_gt, "vol_pred":\
        vol_pred}


def generate_test_metrics_json(gt_dir, pred_dir):
    p_gt_dir = Path(gt_dir)
    p_pred_dir = Path(pred_dir)
    gt_files = sorted(p_gt_dir.rglob("*.nii.gz"))
    pred_files = sorted(p_pred_dir.rglob("*.nii.gz"))
    metric_info = collections.OrderedDict()
    mean_dice = 0
    count = 0
    for pred_file in pred_files:
        for gt_file in gt_files:
            if gt_file.parts[-1] == pred_file.parts[-1]:
                try:
                    info = get_test_metrics(gt_file, pred_file)
                    metric_info[gt_file.as_posix().split("/")[-1]] = info
                    mean_dice += info["dice"]
                    count += 1
                    print("Next File...")
                except RuntimeError:
                    metric_info[gt_file.as_posix().split("/")[-1]] = "NA"
                    continue
    metric_info["mean_dice"] = mean_dice / count
    save_json(metric_info, f"{pred_dir.split('/')[-2]}_metric_info.json")


def ensemble_metric_json(json_file1, json_file2, json_file3):
    with open(json_file1, 'r') as f:
        f1_data = json.load(f)
    with open(json_file2, 'r') as f:
        f2_data = json.load(f)
    with open(json_file3, 'r') as f:
        f3_data = json.load(f)
    mean_dice1 = f1_data.pop("mean_dice")
    mean_dice2 = f2_data.pop("mean_dice")
    mean_dice3 = f3_data.pop("mean_dice")
    mean_dice = (mean_dice1 + mean_dice2 + mean_dice3) / 3
    dictMerged = collections.OrderedDict(f1_data, **f2_data)
    dictMerged = collections.OrderedDict(dictMerged, **f3_data)
    dictMerged["mean_dice"] = mean_dice
    save_json(dictMerged, json_file1.split("_")[0]+json_file2.split("_")[0]+json_file3.split("_")[0]+"_Merged_mertrics_info.json")


import math


def PearsonFirst(X, Y):
    '''
        公式一
    '''
    XY = X * Y
    EX = X.mean()
    EY = Y.mean()
    EX2 = (X ** 2).mean()
    EY2 = (Y ** 2).mean()
    EXY = XY.mean()
    numerator = EXY - EX * EY  # 分子
    denominator = math.sqrt(EX2 - EX ** 2) * math.sqrt(EY2 - EY ** 2)  # 分母

    if denominator == 0:
        return 'NaN'
    rhoXY = numerator / denominator
    return rhoXY


def person_cof_and_diceSTD(json_file):
    with open(json_file, "r") as f:
        f_data = json.load(f)
    vol_gt = []
    vol_pred = []
    dice = []
    for k, v in f_data.items():
        if v != "NA":
            try:
                vol_gt.append(v["vol_gt"])
                vol_pred.append(v["vol_pred"])
                dice.append(v["dice"])
            except TypeError:
                continue
    df = pd.DataFrame({"vol_gt": vol_gt, "vol_pred": vol_pred})

    r = PearsonFirst(df['vol_gt'], df['vol_pred'])  # 使用公式一计算X与Z的相关系数
    print(f"Pearson coefficients: ", r)
    dice_std = np.std(dice, ddof=1)
    print("dice标准差：", dice_std)
    dice_mean = np.mean(dice)
    print("dice均值：", dice_mean)
    # 绘制散点图矩阵

    plt.scatter(vol_gt, vol_pred)
    plt.xlabel("vol_gt")
    plt.ylabel("vol_pred")
    plt.show()


if __name__ == "__main__":
    gt_dir = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task013_carotid/labelsTs"
    pred_dir = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task013_carotid/prediction"
    generate_test_metrics_json(gt_dir, pred_dir)


    # 合并json文件
    json_file1 = "Task001_carotid_metric_info.json"
    json_file2 = "Task002_carotid_metric_info.json"
    json_file3 = "Task003_carotid_metric_info.json"
    ensemble_metric_json(json_file1, json_file2, json_file3)

    # 计算dice方差与相关系数
    person_cof_and_diceSTD("Task007Task008Task009_Merged_mertrics_info.json")
