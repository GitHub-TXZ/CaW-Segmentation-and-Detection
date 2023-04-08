# -*- coding:utf-8 -*-
import os
import json
import shutil
import os
import shutil
from pathlib import Path
import sys
import SimpleITK as sitk
import numpy as np


#
def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)



# json_dict = {}
# json_dict['name'] = "cws"
# json_dict['description'] = "carotid web segmentation"
# json_dict['tensorImageSize'] = "4D"
# json_dict['reference'] = "cws data for nnunet"
# json_dict['licence'] = "dog_shit"
# json_dict['release'] = "0.0"
# json_dict['modality'] = {
#     "0": "CTA",
# }
# json_dict['labels'] = {
#     "0": "background",
#     "1": "carotid web",
# #     "2": "carotid lumen left",
# #     "3": "web right",
# #     "4": "web left"
# }
# # json_dict['numTraining'] = len(cases)
# json_dict['numTraining'] = 57
# json_dict['numTest'] = 4
# json_dict['training'] = [{'image': "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/imagesTr/%s.nii.gz" % i, "label": "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/labelsTr/%s.nii.gz" % i} for i in cases if  int(i.split("_")[-1])<=56]
# # json_dict['training'] = [{'image': "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/imagesTr/%s.nii.gz" % i, "label": "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/labelsTr/%s.nii.gz" % i} for i in cases]
# # json_dict['test'] = [{'image': "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task088_carotid/imagesTs/%s.nii.gz" % i} for i in cases if int(i.split("_")[-1])>=111]
# json_dict['test'] = ["/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/imagesTs/%s.nii.gz" % i for i in cases if int(i.split("_")[-1])>56]
# save_json(json_dict,os.path.join(out, "dataset.json"))



raw = "/homeb/txz/pytorch-UNet/data"
out = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task102_carotid"  # 结构化数据集目录
# cases = subdirs(base, join=False)

maybe_mkdir_p(out)
maybe_mkdir_p(os.path.join(out, "imagesTr"))
maybe_mkdir_p(os.path.join(out, "imagesTs"))
maybe_mkdir_p(os.path.join(out, "labelsTr"))
maybe_mkdir_p(os.path.join(out, "predicton"))

p = Path(raw)
files_im = sorted(p.rglob("*imaging.nii.gz"))
files_seg = sorted(p.rglob("*segmentation.nii.gz"))
count = 0
for file_im, file_seg in zip(files_im, files_seg):
    im = sitk.GetArrayFromImage(sitk.ReadImage(file_im))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(file_seg))
    o = sitk.ReadImage(file_im).GetOrigin()
    d = sitk.ReadImage(file_im).GetDirection()
    s = sitk.ReadImage(file_im).GetSpacing()
    im_patch_web_r = im[236 - 21:236 + 21, 255 - 41:255 + 41, 321 - 41:321 + 41]
    im_patch_web_l = im[236 - 21:236 + 21, 255 - 41:255 + 41, 321 - 41:321 + 41]

    itk_im_patch_r = sitk.GetImageFromArray(im_patch_web_r)
    set_meta(itk_im_patch_r, o, s, d)
    itk_im_patch_l = sitk.GetImageFromArray(im_patch_web_l)
    set_meta(itk_im_patch_l, o, s, d)

    # seg_patch_web_r = seg[240-100:240+100,258-60:258+60,313-100:313+100]
    # seg_patch_web_l = seg[240-100:240+100,258-60:258+60,196-100:196+100]
    seg_patch_web_r = seg[236 - 21:236 + 21, 255 - 41:255 + 41, 321 - 41:321 + 41]
    seg_patch_web_l = seg[236 - 21:236 + 21, 255 - 41:255 + 41, 188 - 41:188 + 41]
    # 把标签变成1
    seg_patch_web_r[seg_patch_web_r < 3] = 0
    seg_patch_web_r[seg_patch_web_r > 4] = 0
    seg_patch_web_r[seg_patch_web_r == 3] = 1
    seg_patch_web_r[seg_patch_web_r == 4] = 1

    seg_patch_web_l[seg_patch_web_l > 4] = 0
    seg_patch_web_l[seg_patch_web_l < 3] = 0
    seg_patch_web_l[seg_patch_web_l == 3] = 1
    seg_patch_web_l[seg_patch_web_l == 4] = 1

    itk_seg_patch_r = sitk.GetImageFromArray(seg_patch_web_r)
    set_meta(itk_seg_patch_r, o, s, d)  # 使用图像的三个信息
    itk_seg_patch_l = sitk.GetImageFromArray(seg_patch_web_l)
    set_meta(itk_seg_patch_l, o, s, d)
    if np.any(seg_patch_web_l):
        sitk.WriteImage(itk_im_patch_l, out + "/imagesTr/%s_0000.nii.gz" % count)
        sitk.WriteImage(itk_seg_patch_l, out + "/labelsTr/%s.nii.gz" % count)
        count += 1

    if np.any(seg_patch_web_r):
        sitk.WriteImage(itk_im_patch_r, out + "/imagesTr/%s_0000.nii.gz" % count)
        sitk.WriteImage(itk_seg_patch_r, out + "/labelsTr/%s.nii.gz" % count)
        count += 1

    print(count - 1)

# 生成json文件
json_dict = {}
json_dict['name'] = "cws"
json_dict['description'] = "carotid web segmentation"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "cws data for nnunet"
json_dict['licence'] = "dog_shit"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CTA",
}
json_dict['labels'] = {
    "0": "background",
    "1": "carotid web",
    #     "2": "carotid lumen left",
    #     "3": "web right",
    #     "4": "web left"
}
# json_dict['numTraining'] = len(cases)
json_dict['numTraining'] = 48
json_dict['numTest'] = 0
json_dict['training'] = [
    {'image': "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task101_carotid/imagesTr/%s.nii.gz" % i,
     "label": "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task101_carotid/labelsTr/%s.nii.gz" % i}
    for i in range(48)]
json_dict['test'] = []
out = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task101_carotid"
save_json(json_dict, os.path.join(out, "dataset.json"))
