# -*- coding: utf-8 -*-

import numpy as np
import os
import SimpleITK as sitk
from medpy import metric
from batchgenerators.utilities.file_and_folder_operations import *
from pathlib import Path
import collections
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from draw_bbox import max_1_connected_domain

join = os.path.join
# 不滑动计算指标
# base = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task777_utimate_proc"


# 滑动计算指标
base = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task888_utimate_proc"
def cal(threshold=50):
    files_test = sorted(Path(join(base, 'labelsTs')).rglob("*.nii.gz"))
    files_prediction = sorted(Path(join(base, 'prediction')).rglob("*.nii.gz"))
    gt_and_pred_web_num_info = {}
    dice = np.array([])
    gt_volume = np.array([])
    pred_volume = np.array([])
    TN, FP, FN, TP = 0, 0, 0, 0
    FP_and_FN_Pat, FP_Pat, FN_Pat = {}, {}, {}
    for file_test, file_prediction in zip(files_test, files_prediction):
        pat_id = file_test.parts[-1]
        I_gt = sitk.ReadImage(file_test)
        I_pred = sitk.ReadImage(file_prediction)
        # 对web结果取最大连通域
        I_pred = max_1_connected_domain(I_pred)

        gt_seg_arr = sitk.GetArrayFromImage(I_gt)
        pred_seg_arr = sitk.GetArrayFromImage(I_pred)
        # print("arr_shape:", gt_seg_arr.shape)
        web_gt = np.sum(gt_seg_arr)
        web_pred = np.sum(pred_seg_arr)
        gt_and_pred_web_num_info[pat_id] = f"gt_web_num:{web_gt},pred_web_num:{web_pred}"
        dice_val = metric.binary.dc(pred_seg_arr, gt_seg_arr)
        # hd = metric.binary.hd95(pred_seg_arr, gt_seg_arr)
        # jc = metric.binary.jc(pred_seg_arr, gt_seg_arr)
        # asd = metric.binary.asd(pred_seg_arr, gt_seg_arr)

        vol_gt = np.sum(gt_seg_arr) * np.prod(I_gt.GetSpacing()) / 1000
        vol_pred = np.sum(pred_seg_arr) * np.prod(I_pred.GetSpacing()) / 1000

        if web_gt != 0:
            dice = np.append(dice, dice_val)
            # 注意如果这里116个数据都用到了（不写到if里面），算出来的相关性可能会偏高
            gt_volume = np.append(gt_volume, vol_gt)
            pred_volume = np.append(pred_volume, vol_pred)
        # web的二分类检测
        if web_gt > 0 and web_pred >= threshold:
            TP += 1
        elif web_gt > 0 and web_pred < threshold:
            FN += 1
            FN_Pat[pat_id] = f"web_gt:{web_gt},web_pred:{web_pred}"
        elif web_gt == 0 and web_pred >= threshold:
            FP += 1
            FP_Pat[pat_id] = f"web_gt:{web_gt},web_pred:{web_pred}"
        else:
            TN += 1
    FP_and_FN_Pat["FN"] = FN_Pat
    FP_and_FN_Pat["FP"] = FP_Pat
    save_json(FP_and_FN_Pat, "FP_and_FN_Pat.json")
    mean_dice = np.mean(dice)
    std_dice = np.std(dice)
    corrcoeff = np.corrcoef(gt_volume, pred_volume)[0][1]
    save_json(gt_and_pred_web_num_info, "gt_and_pred_web_num_info.json")
    ACC = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    fpr = (FP / (FP + TN))
    tpr = (TP / (TP + FN))
    AUC = (fpr * tpr) / 2 + (tpr + 1) * (1 - fpr) / 2
    confusion_matrix = np.array(([TN, FP], [FN, TP]))
    print_str = f"ACC:{ACC},Precision:{Precision},Sensitivity:{Sensitivity},Specificity:{Specificity},AUC:{AUC}"
    print(f"dice:{mean_dice}±{std_dice},corrcoeff:{corrcoeff}")
    print(confusion_matrix)
    print("web detector thresold:", threshold)
    print(print_str)
    print("------------------------------------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    for threshold in range(10, 110, 10):
         cal(threshold)
    # cal(50)
