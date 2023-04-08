from collections import OrderedDict
import os
import SimpleITK
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def copy_Information(srcImage, dstImage):
    dstImage.SetOrigin(srcImage.GetOrigin())
    dstImage.SetDirection(srcImage.GetDirection())
    dstImage.SetSpacing(srcImage.GetSpacing())


if __name__ == "__main__":
    folder = "/media/fabian/My Book/datasets/promise2012"
    out_folder = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task911_脖子上直接分割CaW"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))
    # train

    raw_data = (os.path.join(root, file) for root, dirs, files in
                os.walk('/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/neck_cta') for file in files if
                file.endswith('.nii.gz'))
    segmentations = (os.path.join(root, file) for root, dirs, files in
                     os.walk('/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_seg') for file in files if
                     file.endswith('.nii.gz'))
    for i in raw_data:
        out_fname = join(out_folder, "imagesTr", i.split("/")[-1][:-7] + "_0000.nii.gz")
        sitk.WriteImage(sitk.ReadImage(i), out_fname)
    for i in segmentations:
        out_fname = join(out_folder, "labelsTr", i.split("/")[-1])
        I = sitk.ReadImage(i)
        arr = sitk.GetArrayFromImage(I)[79:508, :, :]
        arr = np.where((arr == 3) | (arr == 4), 1, 0)
        I_ = sitk.GetImageFromArray(arr)
        copy_Information(I, I_)
        sitk.WriteImage(I_, out_fname)
        json_dict = OrderedDict()
        json_dict['name'] = "PROMISE12"
        json_dict['description'] = "prostate"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "see challenge website"
        json_dict['licence'] = "see challenge website"
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "CTA",
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "web"
        }
        json_dict['numTraining'] = 58
        json_dict['numTest'] = 0
        json_dict['test'] = []
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12],
                                  "label": "./labelsTr/%s.nii.gz" % j.split("/")[-1][:-7]} for i, j in
                                 zip(sorted([i for i in os.listdir(out_folder + '/imagesTr')]),
                                     sorted([i for i in os.listdir(out_folder + '/labelsTr')]))]
        save_json(json_dict, os.path.join(out_folder, "dataset.json"))
