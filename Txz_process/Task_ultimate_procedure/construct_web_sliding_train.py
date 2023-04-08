import numpy as np
import SimpleITK as sitk
from pathlib import Path
from medpy import metric
from Txz_process.utils import set_meta
from batchgenerators.utilities.file_and_folder_operations import *


def sliding_proc(cta_arr, seg_arr, stride, pat_name, o, s, d, dir_nnunet):
    if seg_arr.any():
        web_voxel_per_slice = seg_arr.sum(axis=(1, 2))
        cum_sum = np.cumsum(web_voxel_per_slice)
        print(cum_sum, pat_name)
        try:
            upper = np.where(cum_sum >= 10)[0][0]
        except:
            print("这个数据有问题，不读取")
        upper = upper + 1  # 因为区间是左闭右开的区间
        lower = upper - 20
        count = 1
        while True:
            arr_slide = cta_arr[lower:upper, :, :]
            arr_slide_label = seg_arr[lower:upper, :, :]
            arr_slide_itk = sitk.GetImageFromArray(arr_slide)
            arr_slide_label_itk = sitk.GetImageFromArray(arr_slide_label)
            set_meta(arr_slide_itk, o, s, d)
            set_meta(arr_slide_label_itk, o, s, d)
            sitk.WriteImage(arr_slide_itk,
                            dir_nnunet + '/' + "imagesTr/" + pat_name + '_%02d_0000.nii.gz' % count)
            sitk.WriteImage(arr_slide_label_itk,
                            dir_nnunet + '/' + "labelsTr/" + pat_name + '_%02d.nii.gz' % count)
            count += 1
            upper += stride
            lower += stride
            jiankon = seg_arr[lower:upper, :, :].sum(axis=(1, 2)).sum()
            if (jiankon < 10):
                break
    else:
        print("pass")



# 根据每个脉腔的GT画bbox,构造一个web的训练集(不同之处在于这次使用滑动窗口来构建训练集)
def web_trian_gen(raw_cta, raw_seg, dir_nnunet, stride):
    patch_info = {}  # 存储坐标的信息
    p1 = Path(raw_cta)
    p2 = Path(raw_seg)
    ctas = sorted(p1.rglob("*.nii.gz"))
    segs = sorted(p2.rglob("*.nii.gz"))
    total_have_web = 0
    for cta, seg in zip(ctas, segs):
        pat_name = cta.parts[-1][:-7]  # 获取病人的ID:不含有.nii.gz
        cta_itk = sitk.ReadImage(cta)
        cta_arr = sitk.GetArrayFromImage(cta_itk)
        raw_shape = cta_itk.GetSize()  # 保存原始大小
        print(raw_shape)
        o = cta_itk.GetOrigin()
        s = cta_itk.GetSpacing()
        d = cta_itk.GetDirection()
        seg_itk = sitk.ReadImage(seg)
        seg_arr = sitk.GetArrayFromImage(seg_itk)
        seg_arr[seg_arr < 1] = 0
        seg_arr[seg_arr > 4] = 0  # 把脉腔和web以外标签都变成0

        seg_arr[seg_arr == 2] = 1  # 把脉腔标签都变成1

        # 获取三个方向的bbox坐标
        z_min = np.where(np.sum(seg_arr, axis=(1, 2)) != 0)[0][0]
        z_max = np.where(np.sum(seg_arr, axis=(1, 2)) != 0)[0][-1]

        y_min = np.where(np.sum(seg_arr, axis=(0, 2)) != 0)[0][0]
        y_max = np.where(np.sum(seg_arr, axis=(0, 2)) != 0)[0][-1]

        x_min = np.where(np.sum(seg_arr, axis=(0, 1)) != 0)[0][0]
        x_max = np.where(np.sum(seg_arr, axis=(0, 1)) != 0)[0][-1]

        x_mid = (x_min + x_max) // 2  # 将左右脉腔分开的坐标

        new_cta_1 = cta_arr[z_min:z_max, y_min:y_max, x_min:x_mid]  # 第一块脉腔
        new_cta_2 = cta_arr[z_min:z_max, y_min:y_max, x_mid:x_max]  # 第二块脉腔

        new_seg_1 = seg_arr[z_min:z_max, y_min:y_max, x_min:x_mid]  # 注意new_seg_1是来自小的半边x，new_seg_2是来自大的半边x
        new_seg_2 = seg_arr[z_min:z_max, y_min:y_max, x_mid:x_max]

        join = os.path.join
        maybe_mkdir_p(dir_nnunet)
        maybe_mkdir_p(join(dir_nnunet, "imagesTr"))
        maybe_mkdir_p(join(dir_nnunet, "labelsTr"))
        maybe_mkdir_p(join(dir_nnunet, "imagesTs"))
        maybe_mkdir_p(join(dir_nnunet, "labelsTs"))
        maybe_mkdir_p(join(dir_nnunet, "prediction"))

        new_seg_1[new_seg_1 == 1] = 0
        new_seg_1[new_seg_1 == 3] = 1
        new_seg_1[new_seg_1 == 4] = 1
        sliding_proc(new_cta_1, new_seg_1, stride, pat_name + "_1", o, s, d, dir_nnunet)  # 小的半边滑动

        new_seg_2[new_seg_2 == 1] = 0
        new_seg_2[new_seg_2 == 3] = 1
        new_seg_2[new_seg_2 == 4] = 1
        sliding_proc(new_cta_2, new_seg_2, stride, pat_name + "_2", o, s, d, dir_nnunet)  # 大的半边滑动

    # 生成json文件
    p_mydata_cta = Path(join(dir_nnunet, 'imagesTr'))
    p_mydata_seg = Path(join(dir_nnunet, 'labelsTr'))
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
    json_dict['training'] = [{'image': join("imagesTr/", i.parts[-1][:-12] + ".nii.gz"),
                              "label": join("labelsTr/", j.parts[-1][:-7] + ".nii.gz")} for
                             i, j in
                             zip(files_cta1, files_seg)]
    json_dict['test'] = []
    save_json(json_dict, join(dir_nnunet, "dataset.json"))


if __name__ == '__main__':
    stride = 2
    raw_cta = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_cta"
    raw_seg = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_seg"
    dir_nnunet = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task888_utimate_proc"
    web_trian_gen(raw_cta, raw_seg, dir_nnunet, stride)


# 构造训练和验证（很重要）
import numpy as np
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import os
path = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task888_utimate_proc/labelsTr"
dir = listdir(path)
a = load_pickle("/home/txz/PSI-Seg/nnUNet_preprocessed/Task777_utimate_proc/splits_final.pkl")
b = [0]*3
for fold in range(len(a)):
    b[fold] = OrderedDict([("train", np.array([i[:-7] for i in dir if i[:-10] in a[fold]["train"]])),
                       ("val", np.array([(i[:-7])for i in dir if i[:-10] in a[fold]["val"]]))])
