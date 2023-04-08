# -*-coding:utf-8 -*-
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import os
import shutil
from pathlib import Path
import sys
import SimpleITK as sitk
from utils import save_json
from utils import set_meta


# 统计某个数据集中某个标签的slice范围
def range_slice(label_dir_in, label):
    p = Path(label_dir_in)
    files = p.rglob("*seg.nii.gz")
    slice_start = 0
    slice_end = 0
    count = 1
    for file in files:
        print(f"正在读取数据{count}")
        I = sitk.ReadImage(file)
        arr = sitk.GetArrayFromImage(I)
        arr[arr != label] = 0
        z = np.any(arr, axis=(1, 2))
        a = np.where(z)[0][0]
        if slice_start < a:
            slice_start = a
        b = np.where(z)[0][-1]
        if slice_end < b:
            slice_end = b
        count += 1
    print(f"标签GT数据集{label_dir_in}共{count - 1}个样本，标签{label}的最小起始slice为{slice_start}；最大终止slice为{slice_end}")


def get_label_range(file_root, label):
    """
    从一批文件中得到某个标签数据的范围
    Args:
        file_root:数据根路径，eg:r'D:\CTA_Analyze\pytorch-UNet\data'

    Returns:
        z_,x_,y_
    """
    dataset_dir = Path(file_root)
    segs = sorted(dataset_dir.rglob('*seg.nii.gz'))
    x = y = z = 0
    x_ = y_ = z_ = 0
    zmid_list = []
    ymid_list = []
    xmid_list = []
    label_z_min = 999
    label_z_max = 0
    for file in segs:
        # 保留原数据
        itk = sitk.ReadImage(file)
        # 将label以外的标签置0
        arr = sitk.GetArrayFromImage(itk)
        arr[arr != label] = 0
        # 获取单个patch的边界，迭代找到三个axis方向的最大范围
        # Z方向
        zrange = np.sum(arr, axis=(1, 2))
        tempz = np.where(zrange != 0)
        #保存在slice方向上的最大和最小slice
        if tempz[0][-1] > label_z_max:
            label_z_max = tempz[0][-1]
        if tempz[0][0] < label_z_min:
            label_z_min = tempz[0][0]
        if len(tempz[0]) != 0:
            z = tempz[0][-1] - tempz[0][0] + 1  # 跨距
            zmid = (tempz[0][-1] + tempz[0][0]) // 2  # 中心
            if z_ < z:
                z_ = z
        else:
            zmid = 0
        zmid_list.append(zmid)

        # X方向
        xrange = np.sum(arr, axis=(0, 1))
        tempx = np.where(xrange != 0)
        if len(tempx[0]) != 0:
            x = tempx[0][-1] - tempx[0][0] + 1
            xmid = (tempx[0][-1] + tempx[0][0]) // 2
            if x_ < x:
                x_ = x
        else:
            xmid = 0
        xmid_list.append(xmid)

        # Y方向
        yrange = np.sum(arr, axis=(0, 2))
        tempy = np.where(yrange != 0)
        if len(tempy[0]) != 0:
            y = tempy[0][-1] - tempy[0][0] + 1
            ymid = (tempy[0][-1] + tempy[0][0]) // 2
            if y_ < y:
                y_ = y
        else:
            ymid = 0
        ymid_list.append(ymid)

    zmid_list = np.array(zmid_list)
    xmid_list = np.array(xmid_list)
    ymid_list = np.array(ymid_list)
    print(f"数据集合{file_root}中标签{label}的最大跨距z,y,x为{z_, y_, x_}")
    print(label_z_min,label_z_max)
    return z_, y_, x_, np.mean(zmid_list[zmid_list != 0]), np.mean(ymid_list[ymid_list != 0]), np.mean(
        xmid_list[xmid_list != 0])


# 根据中心和最大跨距来确定每个label的patch
def get_patch(path_in, z_, y_, x_, label, path_out, z_extend, y_extend, x_extend):
    """z_extend 代表向两边扩张的slice
       z_ 代表label跨距
    """
    dataset_dir = Path(path_in)
    patch_info = {}  # 准备将裁剪的信息存入到json文件中，方便以后的数据映射
    coordinate = []
    images = sorted(dataset_dir.rglob('*cta1.nii.gz'))
    segs = sorted(dataset_dir.rglob('*seg.nii.gz'))

    for file_seg, file_im in zip(segs, images):
        # os.mkdir("./lumen_{:0>5d}".format(count))
        # os.makedirs("../data_processed2/web_{:0>5d}".format(count))
        itk_label = sitk.ReadImage(file_seg)
        origin = itk_label.GetOrigin()
        spacing = itk_label.GetSpacing()
        direction = itk_label.GetDirection()
        arr_label = sitk.GetArrayFromImage(itk_label)
        arr_label[arr_label != label] = 0
        arr_label[arr_label == label] = 1
        zrange = np.sum(arr_label, axis=(1, 2))
        tempz = np.where(zrange != 0)

        yrange = np.sum(arr_label, axis=(0, 2))
        tempy = np.where(yrange != 0)

        xrange = np.sum(arr_label, axis=(0, 1))
        tempx = np.where(xrange != 0)
        if not os.path.exists(f"./{path_out}/{file_seg.parts[-2]}"):
            os.makedirs(f"./{path_out}/{file_seg.parts[-2]}")

        if len(tempz[0]) == 0:
            patch_info[file_seg.parts[-2]] = f"NA,raw_shape(z,y,x):{arr_label.shape}"
            continue
        else:
            zmid = (tempz[0][0] + tempz[0][-1]) // 2
            ymid = (tempy[0][0] + tempy[0][-1]) // 2
            xmid = (tempx[0][0] + tempx[0][-1]) // 2
            z_start = zmid - z_ // 2 - z_extend
            z_end = zmid + z_ // 2 + z_extend
            y_start = ymid - y_ // 2 - y_extend
            y_end = ymid + y_ // 2 + y_extend
            x_start = xmid - x_ // 2 - x_extend
            x_end = xmid + x_ // 2 + x_extend

            # new_arr_label = arr_label[zmid-z_//2-10:zmid+z_//2+10,xmid-x_//2-30:xmid+x_//2+30,ymid-y_//2-30:ymid+y_//2+30]
            new_arr_label = arr_label[z_start:z_end, y_start:y_end, x_start:x_end]
            patch_info[file_seg.parts[
                -2]] = f"{z_start}:{z_end},{y_start}:{y_end},{x_start}:{x_end},raw_shape(z,y,x):{arr_label.shape}"
            new_itk = sitk.GetImageFromArray(new_arr_label)
            set_meta(new_itk, origin, spacing, direction)

            itk_image = sitk.ReadImage(file_im)
            arr_image = sitk.GetArrayFromImage(itk_image)
            # new_arr_image = arr_image[zmid-z_//2-10:zmid+z_//2+10,xmid-x_//2-30:xmid+x_//2+30,ymid-y_//2-30:ymid+y_//2+30]
            new_arr_image = arr_image[z_start:z_end, y_start:y_end, x_start:x_end]

            new_itk2 = sitk.GetImageFromArray(new_arr_image)
            set_meta(new_itk2, origin, spacing, direction)

            # sitk.WriteImage(new_itk, "./lumen_{:0>5d}".format(count)+'/segmentation.nii.gz')
            # sitk.WriteImage(new_itk2, "./lumen_{:0>5d}".format(count)+'/image.nii.gz')
            sitk.WriteImage(new_itk, f"./{path_out}/{file_seg.parts[-2]}/{file_seg.parts[-2]}_{label}_seg.nii.gz")
            sitk.WriteImage(new_itk2, f"./{path_out}/{file_seg.parts[-2]}/{file_seg.parts[-2]}_{label}_cta1.nii.gz")
    patch_info["label"] = label
    save_json(patch_info, f"{path_out}_patch_info_label{label}.json")


if __name__ == "__main__":
    file_root = "raw_data"
    s = get_label_range(file_root, 2)
    # print(s)
    # get_patch(file_root, s[0], s[1], s[2], 2, "data_patch_carotid_lumen", 10, 10, 10) #后面三个数表示各方向的扩充。s[0],s[1],s[2]表示最大跨距。

# if __name__ == "__main__":
#     label_dir_in = "raw_data"
#     range_slice(label_dir_in, 1)
