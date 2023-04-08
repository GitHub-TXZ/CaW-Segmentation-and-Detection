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


# 完成脖子的截取，统计脉腔的范围
def get_neck_roi(path_in, path_out_cta, path_out_seg):
    dataset_dir = Path(path_in)
    patch_info = {}  # 准备将裁剪的信息存入到json文件中，方便以后的数据映射
    images = sorted(dataset_dir.rglob('*cta1.nii.gz'))
    segs = sorted(dataset_dir.rglob('*seg.nii.gz'))

    for file_seg, file_im in zip(segs, images):
        itk_label = sitk.ReadImage(file_seg)
        origin = itk_label.GetOrigin()
        spacing = itk_label.GetSpacing()
        direction = itk_label.GetDirection()
        arr_label = sitk.GetArrayFromImage(itk_label)
        arr_label[arr_label >= 3] = 0
        arr_label[arr_label == 2] = 1
        patch_info[file_seg.parts[-2]] = f"{file_seg.parts[-1]},raw_shape(z,y,x):{arr_label.shape}"
        new_arr_label = arr_label[79:508, :, :]
        new_itk = sitk.GetImageFromArray(new_arr_label)
        set_meta(new_itk, origin, spacing, direction)

        itk_image = sitk.ReadImage(file_im)
        arr_image = sitk.GetArrayFromImage(itk_image)
        new_arr_image = arr_image[79:508, :, :]
        new_itk2 = sitk.GetImageFromArray(new_arr_image)
        set_meta(new_itk2, origin, spacing, direction)
        sitk.WriteImage(new_itk, path_out_seg + "/" + file_seg.parts[-2] + ".nii.gz")
        sitk.WriteImage(new_itk2, path_out_cta + "/" + file_seg.parts[-2] + ".nii.gz")
        save_json(patch_info, "neck.json")


if __name__ == "__main__":
    path_in = "/homeb/txz/Pycharm_Project/Txz_process/raw_data"
    path_out_cta = "/homeb/txz/Pycharm_Project/Txz_process/Task_ultimate_procedure/neck_cta"
    path_out_seg = "/homeb/txz/Pycharm_Project/Txz_process/Task_ultimate_procedure/neck_seg"
    get_neck_roi(path_in, path_out_cta, path_out_seg)
