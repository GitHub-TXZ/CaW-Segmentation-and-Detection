import numpy as np
import SimpleITK as sitk
from pathlib import Path
from medpy import metric
from Txz_process.utils import set_meta
from two_pass import *
from batchgenerators.utilities.file_and_folder_operations import *

join = os.path.join
from draw_bbox import max_connected_domain

# 给不滑动的送数据
# nnunet_base = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task777_utimate_proc"

# 给滑动的送数据
nnunet_base = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task888_utimate_proc"
raw_cta_path = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_cta"
raw_seg_path = "/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_seg"
web_info = {}
web_center = 0
# web_expend = 20

# expend = 25

base = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task666_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1"
for i in range(3):
    dir = join(base, f"fold_{i}", "validation_raw")
    p = Path(dir)
    pred_segs = sorted(p.rglob("*.nii.gz"))
    for pred_seg in pred_segs:
        pat_id = pred_seg.parts[-1]
        arr1_itk = sitk.ReadImage(pred_seg)
        arr1 = sitk.GetArrayFromImage(arr1_itk)  # 预测的脖子分割
        # 不需要进行最大联通域处理时注释掉下面两行
        arr1_itk = max_connected_domain(arr1_itk)
        arr1 = sitk.GetArrayFromImage(arr1_itk)
        zrange = np.sum(arr1, axis=(1, 2))
        tempz = np.where(zrange != 0)
        upper_z = tempz[0][-1]
        lower_z = tempz[0][0]

        yrange = np.sum(arr1, axis=(0, 2))
        tempy = np.where(yrange != 0)
        upper_y = tempy[0][-1]
        lower_y = tempy[0][0]

        xrange = np.sum(arr1, axis=(0, 1))
        tempx = np.where(xrange != 0)
        upper_x = tempx[0][-1]
        lower_x = tempx[0][0]
        mid_x = (upper_x + lower_x) // 2

        # 在两个连通域基础上算出由于二到四的分界点作为web中心
        # arr1 = arr1[lower_z:upper_z, lower_y:upper_y, lower_x:upper_x]
        # for slice in range(arr1.shape[0]):
        #     graph = two_pass(arr1[slice, :, :])
        #     if len(np.unique(graph)) >= 5:
        #         web_center = slice  # 预处理或得web中心
        #         print("web_center:", web_center)
        #         break

        # 从原始cta和seg中造数据送给nnunet
        I = sitk.ReadImage(join(raw_cta_path, pat_id))
        o = I.GetOrigin()
        s = I.GetSpacing()
        d = I.GetDirection()
        raw_arr = sitk.GetArrayFromImage(I)
        # 完整截取脉腔
        arr_left = raw_arr[79:508, :, :][lower_z:upper_z, lower_y:upper_y, lower_x:mid_x]

        # 连通域分析来估计web中心

        # arr_left = raw_arr[79:508, :, :][lower_z:upper_z, lower_y:upper_y, lower_x:mid_x][
        #            web_center - web_expend:web_center + web_expend, :, :]

        # 只对z轴进行中心化
        # arr_left = raw_arr[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend, lower_y:upper_y, lower_x:mid_x]
        # 使用中心化方法
        # arr_left = raw_arr[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend,
        #            (lower_y + upper_y) // 2 - expend:(lower_y + upper_y) // 2 + expend,
        #            (lower_x + mid_x) // 2 - expend:(lower_x + mid_x) // 2 + expend]
        temp = sitk.GetImageFromArray(arr_left)
        set_meta(temp, o, s, d)
        sitk.WriteImage(temp, join(nnunet_base, f"fold_{i}_test", pat_id[:-7] + "_1_0000.nii.gz"))
        # 完整截取脉腔
        arr_right = raw_arr[79:508, :, :][lower_z:upper_z, lower_y:upper_y, mid_x:upper_x]
        # 连通域分析来估计web中心
        # arr_right = raw_arr[79:508, :, :][lower_z:upper_z, lower_y:upper_y, mid_x:upper_x][
        #             web_center - web_expend:web_center + web_expend, :, :]
        # 只对z轴进行中心化
        # arr_right = raw_arr[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend,
        #            lower_y:upper_y, mid_x:upper_x]
        # 使用中心化方法
        # arr_right = raw_arr[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend,
        #            (lower_y + upper_y) // 2 - expend:(lower_y + upper_y) // 2 + expend,
        #            (upper_x + mid_x) // 2 - expend:(upper_x + mid_x) // 2 + expend]
        temp = sitk.GetImageFromArray(arr_right)
        set_meta(temp, o, s, d)
        sitk.WriteImage(temp, join(nnunet_base, f"fold_{i}_test", pat_id[:-7] + "_2_0000.nii.gz"))

        J = sitk.ReadImage(join(raw_seg_path, pat_id))
        raw_seg = sitk.GetArrayFromImage(J)
        raw_seg[raw_seg > 4] = 0
        raw_seg[raw_seg < 3] = 0
        raw_seg[raw_seg != 0] = 1
        # 完整截取脉腔
        arr_left_seg = raw_seg[79:508, :, :][lower_z:upper_z, lower_y:upper_y, lower_x:mid_x]
        # 连通域分析来估计web中心
        # arr_left_seg = raw_seg[79:508, :, :][lower_z:upper_z, lower_y:upper_y, lower_x:mid_x][
        #                web_center -web_expend:web_center + web_expend, :, :]
        # 只在z轴上进行中心化
        # arr_left_seg = raw_seg[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend,
        #                lower_y:upper_y, lower_x:mid_x]
        # 使用中心化方法
        # arr_left_seg = raw_seg[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend,
        #            (lower_y + upper_y) // 2 - expend:(lower_y + upper_y) // 2 + expend,
        #            (lower_x + mid_x) // 2 - expend:(lower_x + mid_x) // 2 + expend]

        temp = sitk.GetImageFromArray(arr_left_seg)
        set_meta(temp, o, s, d)
        sitk.WriteImage(temp, join(nnunet_base, "labelsTs", pat_id[:-7] + "_1.nii.gz"))

        # 完整截取脉腔
        arr_right_seg = raw_seg[79:508, :, :][lower_z:upper_z, lower_y:upper_y, mid_x:upper_x]
        # 连通域分析来估计web中心
        # arr_right_seg = raw_seg[79:508, :, :][lower_z:upper_z, lower_y:upper_y, mid_x:upper_x][
        #                 web_center - web_expend:web_center + web_expend, :, :]

        # 只在z轴上进行中心化
        # arr_right_seg = raw_seg[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend,
        #                lower_y:upper_y, mid_x:upper_x]
        # 使用中心化方法
        # arr_right_seg = raw_seg[79:508, :, :][(lower_z + upper_z) // 2 - expend:(lower_z + upper_z) // 2 + expend,
        #             (lower_y + upper_y) // 2 - expend:(lower_y + upper_y) // 2 + expend,
        #             (upper_x + mid_x) // 2 - expend:(upper_x + mid_x) // 2 + expend]
        temp = sitk.GetImageFromArray(arr_right_seg)
        set_meta(temp, o, s, d)
        sitk.WriteImage(temp, join(nnunet_base, "labelsTs", pat_id[:-7] + "_2.nii.gz"))

        web1 = np.sum(arr_left_seg)
        web2 = np.sum(arr_right_seg)

        web_info[pat_id[:-7] + "_1.nii.gz"] = str(web1)
        web_info[pat_id[:-7] + "_2.nii.gz"] = str(web2)
save_json(web_info, "web_info.json")  # 能够通过这个文件观察框柱的web是否效果良好
