# coding:utf-8
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def crop_voxels(dir_in):
    """截断像素值"""
    p = Path(dir_in)
    files_cta = sorted(p.rglob("*cta1.nii.gz"))
    files_seg = sorted(p.rglob("*seg.nii.gz"))
    for file_cta, file_seg in zip(files_cta, files_seg):
        arr_cta = sitk.GetArrayFromImage(sitk.ReadImage(file_cta))
        arr_seg = sitk.GetArrayFromImage(sitk.ReadImage(file_seg))
        arr_cta_max = np.max(arr_cta)
        arr_cta_min = np.min(arr_cta)
        arr_cta[arr_cta > min(arr_cta_max, 800)] = min(arr_cta_max, 800)
        arr_cta[arr_cta < max(arr_cta_min, -10)] = max(arr_cta_min, -10)
        itk_cta = sitk.GetImageFromArray(arr_cta)
        itk_cta.CopyInformation(sitk.ReadImage(file_cta))
        sitk.WriteImage(itk_cta, file_cta)
        itk_seg = sitk.GetImageFromArray(arr_seg)
        itk_seg.CopyInformation(sitk.ReadImage(file_seg))
        sitk.WriteImage(itk_seg, file_seg)
        print("next file...")


if __name__ == '__main__':
    dir_in = "/homeb/txz/Pycharm_Project/Txz_process/data_patch_web_small_crop"
    crop_voxels(dir_in)
