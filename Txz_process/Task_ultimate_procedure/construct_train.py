import numpy as np
import SimpleITK as sitk
from pathlib import Path
from medpy import metric
from Txz_process.utils import set_meta
from batchgenerators.utilities.file_and_folder_operations import *


# 读取成对cta与seg
def read_cta_and_seg(cta_path, seg_path):
    p_cta = Path("cta_path")
    p_seg = Path("seg_path")
    cta_files = sorted(p_cta.rglob("*.nii.gz"))
    seg_files = sorted(p_seg.rglob("*.nii.gz"))
    return cta_files, seg_files


# 尝试分别直接在脉腔patch上训练分割模型和滑动构造训练集合
# 将三折交叉验证的信息保存到列表中
def get_fold_info(path_pkl):
    fold_info = load_pickle(path_pkl)
    fold0_train = fold_info[0]['train']
    fold0_val = fold_info[0]['val']
    fold1_train = fold_info[1]['train']
    fold1_val = fold_info[1]['val']
    fold2_train = fold_info[2]['train']
    fold2_val = fold_info[2]['val']
    return fold0_train, fold0_val, fold1_train, fold1_val, fold2_train, fold2_val


# def lumen_patch_seg_web():
#     fold0_train, fold0_val, _, _, _, _ = get_fold_info("splits_final.pkl")
#     cta_files, seg_files = read_cta_and_seg("/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_cta","/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_seg")
#     for file_cta,file_seg in zip(cta_files,seg_files):
#        if file_cta.parts[-1] +


# 根据每个脉腔的GT画bbox,构造一个web的训练集
def web_trian_gen(raw_cta, raw_seg, dir_nnunet):
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
        cta_itk1 = sitk.GetImageFromArray(new_cta_1)
        cta_itk2 = sitk.GetImageFromArray(new_cta_2)
        new_seg_1 = seg_arr[z_min:z_max, y_min:y_max, x_min:x_mid]  # 注意new_seg_1是来自小的半边x，new_seg_2是来自大的半边x
        new_seg_2 = seg_arr[z_min:z_max, y_min:y_max, x_mid:x_max]

        set_meta(cta_itk1, o, s, d)
        set_meta(cta_itk2, o, s, d)

        join = os.path.join
        maybe_mkdir_p(dir_nnunet)
        maybe_mkdir_p(join(dir_nnunet, "imagesTr"))
        maybe_mkdir_p(join(dir_nnunet, "labelsTr"))
        maybe_mkdir_p(join(dir_nnunet, "imagesTs"))
        maybe_mkdir_p(join(dir_nnunet, "labelsTs"))
        maybe_mkdir_p(join(dir_nnunet, "prediction"))
        sitk.WriteImage(cta_itk1, join(dir_nnunet, "imagesTr", pat_name + "_1_0000.nii.gz"))
        sitk.WriteImage(cta_itk2, join(dir_nnunet, "imagesTr", pat_name + "_2_0000.nii.gz"))
        new_seg_1[new_seg_1 == 1] = 0
        new_seg_1[new_seg_1 == 3] = 1
        new_seg_1[new_seg_1 == 4] = 1

        new_seg_2[new_seg_2 == 1] = 0
        new_seg_2[new_seg_2 == 3] = 1
        new_seg_2[new_seg_2 == 4] = 1

        seg_itk_1 = sitk.GetImageFromArray(new_seg_1)
        seg_itk_2 = sitk.GetImageFromArray(new_seg_2)

        set_meta(seg_itk_1, o, s, d)
        set_meta(seg_itk_2, o, s, d)
        web_1 = np.sum(new_seg_1)
        web_2 = np.sum(new_seg_2)

        total_have_web = total_have_web + (web_1 != 0) + (web_2 != 0)

        patch_info[
            pat_name] = f"{z_min}:{z_max},{y_min}:{y_max},{x_min}:{x_mid}:{x_max},raw_shape(z,y,x):{seg_arr.shape},web_num:{pat_name}_1:{web_1},{pat_name}_2:{web_2}"  # 保存坐标信息以及web的数量信息

        sitk.WriteImage(seg_itk_1, join(dir_nnunet, 'labelsTr', pat_name + "_1.nii.gz"))
        sitk.WriteImage(seg_itk_2, join(dir_nnunet, 'labelsTr', pat_name + "_2.nii.gz"))

    patch_info['total_have_web'] = str(total_have_web)
    save_json(patch_info, "patch_info.json")
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
    raw_cta = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_cta"
    raw_seg = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_seg"
    dir_nnunet = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task777_utimate_proc"
    web_trian_gen(raw_cta, raw_seg, dir_nnunet)

# 修改划分文件代码
# from collections import OrderedDict
#
# b = [0]*3
# for i in range(len(a)):
#     b[i] = OrderedDict([("train", np.array([k + f'_{j}' for j in range(1, 3) for k in a[i]["train"]])),
#                         ("val", np.array([k + f'_{j}' for j in range(1, 3) for k in a[i]["val"]]))])
