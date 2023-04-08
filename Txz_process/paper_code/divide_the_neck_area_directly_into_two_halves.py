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
    out_folder = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task912_脖子劈成两半直接分割CaW"

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
        out_fname_1 = join(out_folder, "imagesTr", i.split("/")[-1][:-7] + "_1_0000.nii.gz")
        out_fname_2 = join(out_folder, "imagesTr", i.split("/")[-1][:-7] + "_2_0000.nii.gz")
        itk = sitk.ReadImage(i)
        arr = sitk.GetArrayFromImage(itk)
        arr1 = arr[:, :, 0:256]
        arr2 = arr[:, :, 256:512]

        itk1 = sitk.GetImageFromArray(arr1)
        itk2 = sitk.GetImageFromArray(arr2)
        copy_Information(itk, itk1)
        copy_Information(itk, itk2)
        sitk.WriteImage(itk1, out_fname_1)
        sitk.WriteImage(itk2, out_fname_2)
    for i in segmentations:
        out_fname_1 = join(out_folder, "labelsTr", i.split("/")[-1][:-7] + "_1.nii.gz")
        out_fname_2 = join(out_folder, "labelsTr", i.split("/")[-1][:-7] + "_2.nii.gz")
        itk = sitk.ReadImage(i)
        arr = sitk.GetArrayFromImage(itk)[79:508, :, :]
        arr = np.where((arr == 3) | (arr == 4), 1, 0)
        arr1 = arr[:, :, 0:256]
        arr2 = arr[:, :, 256:512]
        # a = np.sum(arr1)
        # b = np.sum(arr2)
        itk1 = sitk.GetImageFromArray(arr1)
        itk2 = sitk.GetImageFromArray(arr2)
        copy_Information(itk, itk1)
        copy_Information(itk, itk2)
        sitk.WriteImage(itk1, out_fname_1)
        sitk.WriteImage(itk2, out_fname_2)
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
        json_dict['numTraining'] = 116
        json_dict['numTest'] = 0
        json_dict['test'] = []
        json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12],
                                  "label": "./labelsTr/%s.nii.gz" % j.split("/")[-1][:-7]} for i, j in
                                 zip(sorted([i for i in os.listdir(out_folder + '/imagesTr')]),
                                     sorted([i for i in os.listdir(out_folder + '/labelsTr')]))]
        save_json(json_dict, os.path.join(out_folder, "dataset.json"))



# 下面的代码用于更改splits_final文件
import os
import SimpleITK as sitk
import numpy as np

# set the directory path
dir_path = '/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task912_脖子劈成两半直接分割CaW/labelsTr'

# create an empty list to store the file names
file_list = []

# loop through all files in the directory
for file in os.listdir(dir_path):
    # check if the file is a .nii.gz file
    if file.endswith('.nii.gz'):
        # read the file and check if it contains non-zero values
        img = sitk.ReadImage(os.path.join(dir_path, file))
        arr = sitk.GetArrayFromImage(img)
        if np.sum(arr) != 0:
            # if the file contains non-zero values, add its name to the list
            file_list.append(file[:-7])

# print the list of file names
print(len(file_list))
from batchgenerators.utilities.file_and_folder_operations import *
x = load_pickle("/home/txz/PSI-Seg/nnUNet_preprocessed/Task777_utimate_proc/splits_final.pkl")
# loop through each dictionary in the list
for d in x:
    # get the 'train' array from the dictionary
    train_arr = d['train']
    # create a new array with only the elements that are in file_list
    new_train_arr = np.array([elem for elem in train_arr if elem in file_list])
    # update the 'train' key in the dictionary with the new array
    d['train'] = new_train_arr
folder = "/home/txz/PSI-Seg/nnUNet_preprocessed/Task912_脖子劈成两半直接分割CaW"
from os.path import join
save_pickle(x, join(folder, "splits_final.pkl"))
print(x)
