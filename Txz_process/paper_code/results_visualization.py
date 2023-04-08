# -*- coding:utf-8 -*-
import cv2
import SimpleITK as sitk
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from remap_back_to_original_size import *
from medpy.metric.binary import dc


# from medpy import dc
# array = sitk.GetArrayFromImage(
#     sitk.ReadImage(r"D:\Pycharm_Project\CTA_Seg\Txz_process\raw_data\Doubt_01-029\Doubt_01-029_seg.nii.gz"))
# array = array[:, 256, :].astype(np.uint8)
# a, b = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def show_all():
    array = sitk.GetArrayFromImage(sitk.ReadImage(ct2_path))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(gt2_path))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(ours2))
    slice_ = 257
    # slice_ = 26
    print(dc(pred[:, slice_, :], mask[:, slice_, :]))
    mask = mask[:, slice_, :].astype(np.uint8)
    pred = pred[:, slice_, :].astype(np.uint8)
    img = array[:, slice_, :][:, :, None].repeat(3, axis=2)
    img = exposure.rescale_intensity(img, in_range=(-50, 600), out_range=(0, 1))
    # 因为有上下翻转
    mask = mask[::-1, :]
    pred = pred[::-1, :]
    pos_list, pos_info_list = get_pos(mask)
    pos_list1, pos_info_list1 = get_pos(pred)
    fig, ax = plt.subplots()
    # 因为有上下翻转
    img = img[::-1, :]
    ax.imshow(img, cmap='gray')
    for pos in pos_list1:  # pred
        ax.fill(pos[0], pos[1], facecolor='#AC2B2C', edgecolor='none', alpha=1)
        # ax.fill(pos[0], pos[1], facecolor='none', edgecolor='r', alpha=1)
    for pos in pos_list:  # gt
        ax.fill(pos[0], pos[1], facecolor='none', edgecolor='g', alpha=1)
    ax.axis('off')
    plt.savefig('/home/txz/PSI-Seg/out.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def get_pos(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pos_info = []
    pos_list = []
    for i in range(len(contours)):
        pos = np.array(contours[i]).squeeze(1)
        x = []
        y = []
        for a in pos:
            x.append(a[0])
            y.append(a[1])
        pos_list.append((x, y))
        info = {'pos': pos, 'hierarchy': hierarchy[0][i]}
        pos_info.append(info)
    return pos_list, pos_info


# 可视化脉腔分割
ct2_path = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task666_utimate_proc_脉腔不滑动/imagesTr/Doubt_01-029_0000.nii.gz"
gt2_path = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task666_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/Doubt_01-029.nii.gz"
ours2 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task666_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/Doubt_01-029.nii.gz"
show_all()

# 可视化web分割
# ct2_path = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task777_utimate_proc_web在脉腔上不滑动/imagesTr/Doubt_01-029_1_0000.nii.gz"
# gt2_path = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task777_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/gt_niftis/Doubt_01-029_1.nii.gz"
# ours2 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task777_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/Doubt_01-029_1.nii.gz"
# show_all()
