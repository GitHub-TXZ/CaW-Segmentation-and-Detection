import numpy as np
import SimpleITK as sitk
from medpy import metric

try:
    from batchgenerators.utilities.file_and_folder_operations import *
except:
    pass


# 进行连通域分析， 获取最大的两个连通域
def max_connected_domain(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数

    area_max_label = 0  # 最大的连通域的label
    area_max = 0

    second_max_label = 0
    second_max_area = 0

    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > area_max:
            area_max_label = i
            area_max = area

    temp = list(range(1, num_connected_label + 1))
    temp.remove(area_max_label)
    for i in temp:
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > second_max_area:
            second_max_label = i
            second_max_area = area

    np_output_mask = sitk.GetArrayFromImage(output_mask)

    res_mask = np.zeros_like(np_output_mask)
    res_mask[np_output_mask == area_max_label] = 1
    res_mask[np_output_mask == second_max_label] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())

    return res_itk


# 快速获取两个文件的DSC
def fast_get_dice(gt_label_path, pred_label_path):
    i1 = sitk.ReadImage(gt_label_path)
    gt_arr = sitk.GetArrayFromImage(i1)
    i2 = sitk.ReadImage(pred_label_path)
    pred_arr = sitk.GetArrayFromImage(i2)
    dice = metric.binary.dc(pred_arr, gt_arr)
    print("dice:", dice)


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


# 根据后处理后的预测文件画一个框框,将左右脉腔分开
def get_bbox(pred_seg_file, cta_file):
    I1 = sitk.ReadImage(pred_seg_file)
    I2 = sitk.ReadImage(cta_file)
    arr1 = sitk.GetArrayFromImage(I1)
    arr2 = sitk.GetArrayFromImage(I2)
    print("raw shape：", arr1.shape)
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
    print("after bounding box shape", upper_z - lower_z, upper_y - lower_y, upper_x - lower_x)
    arr_left = arr1[lower_z:upper_z, lower_y:upper_y, lower_x:mid_x]

    itk_left = sitk.GetImageFromArray(arr_left)
    itk_left.SetOrigin(I1.GetOrigin())
    itk_left.SetSpacing(I1.GetSpacing())
    itk_left.SetDirection(I1.GetDirection())
    sitk.WriteImage(itk_left, "Doubt_06-017_post_2.nii.gz")

    arr_right = arr1[lower_z:upper_z, lower_y:upper_y, mid_x:upper_x]
    itk_right = sitk.GetImageFromArray(arr_right)
    itk_right.SetOrigin(I1.GetOrigin())
    itk_right.SetSpacing(I1.GetSpacing())
    itk_right.SetDirection(I1.GetDirection())
    sitk.WriteImage(itk_right, "Doubt_06-017_post_1.nii.gz")

    # 对脖子上的cta图像执行同样的操作，如果有需要请记录x，y，z的截断坐标，后续map回去原图
    arr2_left = arr2[lower_z:upper_z, lower_y:upper_y, lower_x:mid_x]
    a = sitk.GetImageFromArray(arr2_left)
    a.CopyInformation(itk_left)
    sitk.WriteImage(a, "Doubt_06-017_cta_2.nii.gz")

    arr2_right = arr2[lower_z:upper_z, lower_y:upper_y, mid_x:upper_x]
    b = sitk.GetImageFromArray(arr2_right)
    b.CopyInformation(itk_right)
    sitk.WriteImage(b, "Doubt_06-017_cta_1.nii.gz")


def max_1_connected_domain(itk_mask):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数,注意这里可能是为0的
    # print(num_connected_label)

    area_max_label = 0  # 最大的连通域的label
    area_max = 0

    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > area_max:
            area_max_label = i
            area_max = area

    np_output_mask = sitk.GetArrayFromImage(output_mask)

    res_mask = np.zeros_like(np_output_mask)
    if num_connected_label !=0:
        res_mask[np_output_mask == area_max_label] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())

    return res_itk
if __name__ == '__main__':
    # itk_mask = sitk.ReadImage("./Doubt_06-017.nii.gz")
    # itk = max_connected_domain(itk_mask)
    # sitk.WriteImage(itk, "./Doubt_06-017_post.nii.gz")
    # fast_get_dice("Doubt_06-017_GT.nii.gz", "Doubt_06-017_post.nii.gz")
    # print(get_fold_info("splits_final.pkl"))
    get_bbox("Doubt_06-017_post.nii.gz", "../Task_ultimate_procedure/neck_cta/Doubt_06-017_cta.nii.gz")
