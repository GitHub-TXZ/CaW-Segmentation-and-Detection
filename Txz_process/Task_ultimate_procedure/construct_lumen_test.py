import numpy as np
import SimpleITK as sitk
from pathlib import Path
from medpy import metric
from Txz_process.utils import set_meta
import os
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
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

join = os.path.join
from draw_bbox import max_connected_domain

# 构造测试数据
# nnunet_base = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task555_utimate_proc"
# raw_cta_path = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/neck_cta"
# raw_seg_path = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/neck_seg"
# f0 = join(nnunet_base, "fold_0_test")
# maybe_mkdir_p(f0)
#
# f1 = join(nnunet_base, "fold_1_test")
# maybe_mkdir_p(f1)
#
# f2 = join(nnunet_base, "fold_2_test")
# maybe_mkdir_p(join(f2))
#
# aa = [0] * 3
# a = load_pickle("/home/txz/PSI-Seg/nnUNet_preprocessed/Task666_utimate_proc/splits_final.pkl")
# aa[0] = a[0]["val"]
# aa[1] = a[1]["val"]
# aa[2] = a[2]["val"]
# for i in aa[0]:
#     shutil.copy(join(raw_cta_path, i + ".nii.gz"), join(f0, i + "_0000.nii.gz"))
# for i in aa[1]:
#     shutil.copy(join(raw_cta_path, i + ".nii.gz"), join(f1, i + "_0000.nii.gz"))
# for i in aa[2]:
#     shutil.copy(join(raw_cta_path, i + ".nii.gz"), join(f2, i + "_0000.nii.gz"))
# #
# 计算在脖子上滑动脉腔的测试指标
base = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task555_utimate_proc"
files_test = sorted(Path(join(base, 'labelsTs')).rglob("*.nii.gz"))
files_prediction = sorted(Path(join(base, 'prediction')).rglob("*.nii.gz"))
dice = np.array([])
gt_volume = np.array([])
pred_volume = np.array([])
for file_test, file_prediction in zip(files_test, files_prediction):
    pat_id = file_test.parts[-1]
    I_gt = sitk.ReadImage(file_test)
    I_pred = sitk.ReadImage(file_prediction)
    gt_seg_arr = sitk.GetArrayFromImage(I_gt)
    pred_seg_arr = sitk.GetArrayFromImage(I_pred)
    # print("arr_shape:", gt_seg_arr.shape)
    dice_val = metric.binary.dc(pred_seg_arr, gt_seg_arr)
    print(dice_val)
    dice = np.append(dice, dice_val)
    # hd = metric.binary.hd95(pred_seg_arr, gt_seg_arr)
    # jc = metric.binary.jc(pred_seg_arr, gt_seg_arr)
    # asd = metric.binary.asd(pred_seg_arr, gt_seg_arr)

    vol_gt = np.sum(gt_seg_arr) * np.prod(I_gt.GetSpacing()) / 1000
    vol_pred = np.sum(pred_seg_arr) * np.prod(I_pred.GetSpacing()) / 1000
    gt_volume = np.append(gt_volume,vol_gt)
    pred_volume = np.append(pred_volume,vol_pred)

mean_dice = np.mean(dice)
std_dice = np.std(dice)
corrcoeff = np.corrcoef(gt_volume, pred_volume)[0][1]

print(f"dice:{mean_dice}±{std_dice},corrcoeff:{corrcoeff}")
