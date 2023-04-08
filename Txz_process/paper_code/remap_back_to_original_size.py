import SimpleITK as sitk
import numpy as np


def get_fantu():
    lumen_patch = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task666_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/Doubt_01-029.nii.gz"
    web_patch_1 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task777_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/Doubt_01-029_1.nii.gz"
    web_patch_2 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task777_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/Doubt_01-029_2.nii.gz"
    arr1 = sitk.GetArrayFromImage(sitk.ReadImage(web_patch_1))
    arr1[arr1 == 1] = 3
    arr2 = sitk.GetArrayFromImage(sitk.ReadImage(web_patch_2))
    arr2[arr2 == 1] = 4

    lumen_fantu = sitk.GetArrayFromImage(sitk.ReadImage(lumen_patch))
    lumen_fantu[:, :, :256][lumen_fantu[:, :, :256] == 1] = 1
    lumen_fantu[:, :, 256:][lumen_fantu[:, :, 256:] == 1] = 2

    # 第三个维度小于256的置1，大于256的置2，返回左右标签

    web_fantu = np.concatenate((arr1, arr2), axis=2)
    # web_fantu[web_fantu == 1] = 2

    total_fantu = np.zeros([743, 512, 512])
    total_fantu[79:508] = lumen_fantu
    # 下面这种写法是错误的，因为我的三个patch info坐标是直接来源于全图取值
    # total_fantu[79:508][292:444,227:285,197:372][web_fantu == 2] = 2
    total_fantu[292:444, 227:285, 197:372][web_fantu == 3] = 3
    total_fantu[292:444, 227:285, 197:372][web_fantu == 4] = 4

    total_fantu = total_fantu.astype(np.uint8)
    return total_fantu


import SimpleITK as sitk

fantu = get_fantu()
target_img = sitk.GetImageFromArray(fantu)
# 读取源图片和目标图片
source_img = sitk.ReadImage("/home/txz/PSI-Seg/Txz_process/raw_data/Doubt_01-029/cta1.nii.gz")

# 复制源图片的元信息给目标图片
target_img.CopyInformation(source_img)

# 保存目标图片
sitk.WriteImage(target_img, '/home/txz/PSI-Seg/Doubt_01-029_prediction.nii.gz')

gt_seg = "/home/txz/PSI-Seg/Txz_process/raw_data/Doubt_01-029/Doubt_01-029_seg.nii.gz"
gt_Image = sitk.ReadImage(gt_seg)
gt_seg_arr = sitk.GetArrayFromImage(gt_Image)
gt_seg_arr[(gt_seg_arr != 1) & (gt_seg_arr != 2) & (gt_seg_arr != 3) & (gt_seg_arr != 4)] = 0  # 将不等于1、2、3、4的标签值置0
final_gt_Image = sitk.GetImageFromArray(gt_seg_arr)
final_gt_Image.CopyInformation(gt_Image)

sitk.WriteImage(final_gt_Image, "/home/txz/PSI-Seg/Doubt_01-029_ground_truth.nii.gz")
