import os
import SimpleITK as sitk
import numpy as np

def has_label_one(seg_path):
    files = os.listdir(seg_path)
    result = {}
    for file in sorted(files):
        if not file.endswith('.nii.gz'):
            continue
        seg = sitk.ReadImage(os.path.join(seg_path, file))
        seg_arr = sitk.GetArrayFromImage(seg)
        if 1 in seg_arr:
            result[file] = True
        else:
            result[file] = False
    return result

result = has_label_one("/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task999_bifurcation_proc/labelsTr")
print(result)