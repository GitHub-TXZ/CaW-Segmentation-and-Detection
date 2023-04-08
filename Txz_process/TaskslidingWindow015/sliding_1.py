# -*- coding:utf-8 -*-
import pickle

import shutil

import os

from batchgenerators.utilities.file_and_folder_operations import load_json
from pathlib import Path
import json
import SimpleITK as sitk
import re
from Txz_process.utils import set_meta
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

# print("************************************")
# # # 记录116个patch中有哪些是存在web的，记录在json文件中 67个
# p = Path(
#     "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/GT_seg_maybe_include_web")
# files = p.rglob("*.nii.gz")
# num = 0
# file_name = []
# for file in files:
#     i = sitk.ReadImage(file)
#     arr = sitk.GetArrayFromImage(i)
#     if np.any(arr):
#         if arr.sum(axis=(1, 2)).sum() > 10:
#             print(arr.sum(axis=(1, 2)).sum())
#             num += 1
#             file_name.append(file.parts[-1])
# with open("/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/have_web.json",
#           "w") as f:
#     json.dump(file_name, f)
# print(num)
#
# # 获取第一阶段116个脉腔的预测结果的上下限作为 web上下滑动窗口的限制信息,以及中心z坐标
# p = Path(
#     "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/Prediction_carotid_lumen_seg")
# files = p.rglob("*.nii.gz")
# slide_limit = {}
# num = 0
# for file in files:
#     i = sitk.ReadImage(file)
#     arr = sitk.GetArrayFromImage(i)
#     slice_sum = arr.sum(axis=(1, 2))
#     temp = np.where(slice_sum != 0)
#     lower = int(temp[0][0])
#     upper = int(temp[0][-1])
#     center = int((upper + lower) // 2)
#     slide_limit[file.parts[-1]] = (lower, upper, center)
#     num += 1
#     print(num)
# with open("/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/slide_limit.json",
#           "w") as f:
#     json.dump(slide_limit, f)


# 统计web的体素信息，原始数据中一共有39个数据含有label3，有22个数据含有label4，但是在116个patch的数据集中有69个有web，\ 其中67个超过10个体素，2个小于10个体素
# 因为左patch和右边patch在web上可能存在交集
# p = Path("/homeb/txz/Pycharm_Project/Txz_process/raw_data")
# files = p.rglob("*seg.nii.gz")
# num3 = 0
# num4 = 0
# label_info = {}
# lable3 = []
# label4 = []
# for file in files:
#     i = sitk.ReadImage(file)
#     arr = sitk.GetArrayFromImage(i)
#     if np.any(arr == 3):
#         lable3.append(np.sum(arr == 3))
#         num3 += 1
#     if np.any(arr == 4):
#         label4.append(np.sum(arr == 4))
#         num4 += 1
# label_info["label3"] = lable3
# label_info["label4"] = label4
# print("label3有%d个" % num3)
# print("label4有%d个" % num4)


# # 把没有web的提取出来也送进网络做个滑动窗口的检测,在一共获取的59个样本中有web的只有10个，方便二分类检测任务
# src_cta = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/carotid_lumen_patch"
# src_seg = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/GT_seg_maybe_include_web"
# dst_cta = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Separate_for_test_cta"
# dst_seg = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Separate _for_test_seg"
# have_web_path = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/have_web.json"
# have_web = load_json(have_web_path)
# all_cta = os.listdir(
#     "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/carotid_lumen_patch")
# all_seg = os.listdir(
#     "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/GT_seg_maybe_include_web")
#
# for i, j in zip(all_cta, all_seg):
#     if i[:-12] + ".nii.gz" not in have_web:
#         shutil.copy(os.path.join(src_cta, i), os.path.join(dst_cta, i))
#         shutil.copy(os.path.join(src_seg, j), os.path.join(dst_seg, j))

# 进行二分类web检测任务
# from pathlib import Path
# import SimpleITK as sitk
# import numpy as np
#
# GT_seg_path = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Separate _for_test_seg"
# Pred_seg_path = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Prediction_fold_2"
# GT_files = sorted(Path(GT_seg_path).rglob("*.nii.gz"))
# Pred_files = sorted(Path(Pred_seg_path).rglob("*.nii.gz"))
# pred_have_web = []
# pred_not_have_web = []
# with open("/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/ten_have_web.pkl",
#           "rb") as f:
#     ten_have_web = pickle.load(f)
# for Pred_file in Pred_files:
#     pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(Pred_file))
#     if np.sum(pred_arr, axis=(1, 2)).sum() >= 10:
#         pred_have_web.append(Pred_file.parts[-1])
#     else:
#         pred_not_have_web.append(Pred_file.parts[-1])
# confusion_matrix = np.zeros((2, 2))
# for i in pred_not_have_web:
#     if i in ten_have_web:
#         confusion_matrix[1][0] += 1
#     else:
#         confusion_matrix[0][0] += 1
# for j in pred_have_web:
#     if j in ten_have_web:
#         confusion_matrix[1][1] += 1
#     else:
#         confusion_matrix[0][1] += 1
# print("confusion_martix:", confusion_matrix)


class siliding_window_web():
    def __init__(self, ):
        pass

    def get_GT(self, dir_in, dir_out, jsonfile_label1, jsonfile_label2):
        p = Path(dir_in)
        with open(jsonfile_label1, "rb") as f:
            coordinate1 = json.load(f)
        with open(jsonfile_label2, "rb") as f:
            coordinate2 = json.load(f)
        files_GT_seg = sorted(p.rglob("*seg.nii.gz"))
        for file_GT_seg in files_GT_seg:
            raw_itk = sitk.ReadImage(file_GT_seg)
            o = raw_itk.GetOrigin()
            s = raw_itk.GetSpacing()
            d = raw_itk.GetDirection()
            gt_seg_arr = sitk.GetArrayFromImage(raw_itk)
            gt_seg_arr[gt_seg_arr == 1] = 0
            gt_seg_arr[(gt_seg_arr == 3) | (gt_seg_arr == 4)] = 1
            gt_seg_arr[gt_seg_arr != 1] = 0
            pat_id = file_GT_seg.parts[-2]
            label1_coor = re.findall("\d+\.?\d*", coordinate1[pat_id])
            label1_coor = list(map(int, label1_coor))
            label1_new_seg = gt_seg_arr[label1_coor[0]:label1_coor[1], label1_coor[2]:label1_coor[3],
                             label1_coor[4]:label1_coor[5]]

            i1 = sitk.GetImageFromArray(label1_new_seg)
            set_meta(i1, o, s, d)
            # i1.CopyInformation(sitk.ReadImage(file_GT_seg)) 不知道为什么有问题
            sitk.WriteImage(i1, dir_out + "/" + pat_id + "_1.nii.gz")

            label2_coor = re.findall("\d+\.?\d*", coordinate2[pat_id])
            label2_coor = list(map(int, label2_coor))
            label2_new_seg = gt_seg_arr[label2_coor[0]:label2_coor[1], label2_coor[2]:label2_coor[3], \
                             label2_coor[4]:label2_coor[5]]
            i2 = sitk.GetImageFromArray(label2_new_seg)
            set_meta(i2, o, s, d)
            # i2.CopyInformation(sitk.ReadImage(file_GT_seg))
            sitk.WriteImage(i2, dir_out + "/" + pat_id + "_2.nii.gz")

    def sliding_web_patch_gen(self, carotid_lumen_patch, GT_seg_maybe_include_web, Prediction_carotid_lumen_seg,
                              gen_out_dir_cta, gen_out_dir_label, stride):
        p1 = Path(carotid_lumen_patch)
        p2 = Path(GT_seg_maybe_include_web)
        p3 = Path(Prediction_carotid_lumen_seg)
        i = sorted(p1.rglob("*.nii.gz"))
        j = sorted(p2.rglob("*.nii.gz"))
        k = sorted(p3.rglob("*.nii.gz"))
        a = 0
        b = 0
        f1 = load_pickle(
            "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/30_have_web.pkl")
        have_web = load_json(
            "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/have_web.json")
        for ids, jds, kds in zip(i, j, k):
            if (jds.parts[-1] not in f1) and (jds.parts[-1] in have_web):
                print(jds.parts[-1])
                itk1 = sitk.ReadImage(ids)
                o = itk1.GetOrigin()
                s = itk1.GetSpacing()
                d = itk1.GetDirection()
                itk2 = sitk.ReadImage(jds)
                itk3 = sitk.ReadImage(kds)
                arr1 = sitk.GetArrayFromImage(itk1)  # carotid_lumen_patch
                arr2 = sitk.GetArrayFromImage(itk2)  # GT_seg_maybe_include_web
                # arr3 = sitk.GetArrayFromImage(itk3)  # Prediction_carotid_lumen_seg
                if arr2.any():
                    web_voxel_per_slice = arr2.sum(axis=(1, 2))
                    cum_sum = np.cumsum(web_voxel_per_slice)
                    print(cum_sum, ids.parts[-1])
                    try:
                        upper = np.where(cum_sum >= 10)[0][0]
                    except:
                        print("这个数据有问题，不读取")
                        continue
                    upper = upper + 1  # 因为区间是左闭右开的区间
                    lower = upper - 20
                    count = 1
                    while True:
                        arr_slide = arr1[lower:upper, :, :]
                        arr_slide_label = arr2[lower:upper, :, :]
                        arr_slide_itk = sitk.GetImageFromArray(arr_slide)
                        arr_slide_label_itk = sitk.GetImageFromArray(arr_slide_label)
                        set_meta(arr_slide_itk, o, s, d)
                        set_meta(arr_slide_label_itk, o, s, d)
                        sitk.WriteImage(arr_slide_itk,
                                        gen_out_dir_cta + '/' + ids.parts[-1][:-11] + '%d_0000.nii.gz' % count)
                        sitk.WriteImage(arr_slide_label_itk,
                                        gen_out_dir_label + '/' + jds.parts[-1][:-7] + '_%d.nii.gz' % count)
                        b += 1
                        count += 1
                        upper += stride
                        lower += stride
                        jiankon = arr2[lower:upper, :, :].sum(axis=(1, 2)).sum()
                        if (jiankon < 10):
                            count = 1
                            break
                else:
                    continue

            else:
                shutil.copy(ids, "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/test/" +
                            ids.parts[-1])
                shutil.copy(jds, "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu"
                                 "/label_test/" + jds.parts[-1])
                a += 1
        print("have %d test cases" % a)
        print("total have %d generate in sliding" % b)


if __name__ == '__main__':
    a = siliding_window_web()
    # dir_in = "/homeb/txz/Pycharm_Project/Txz_process/raw_data"
    # dir_out = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/TaskSlidingWindow015/GT_seg_maybe_include_web"
    # jsonfile_label1 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/TaskSlidingWindow015/data_patch_carotid_lumen_patch_info_label1.json"
    # jsonfile_label2 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/TaskSlidingWindow015/data_patch_carotid_lumen_patch_info_label2.json"
    # a.get_GT(dir_in, dir_out, jsonfile_label1, jsonfile_label2)
    carotid_lumen_patch = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/carotid_lumen_patch"
    GT_seg_include_web = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web" \
                         "/GT_seg_maybe_include_web"
    gen_out_dir_cta = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/imagesTr"
    gen_out_dir_label = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/labelsTr"
    Prediction_carotid_lumen_seg = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data" \
                                   "/Task015_sliding_for_web/Prediction_carotid_lumen_seg"
    stride = 2  # 步长为2
    a.sliding_web_patch_gen(carotid_lumen_patch, GT_seg_include_web, Prediction_carotid_lumen_seg, gen_out_dir_cta,
                            gen_out_dir_label, stride)
