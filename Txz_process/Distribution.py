# coding:utf-8
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def cal_iqr(arr):  # 计算四分位距
    qr1 = np.quantile(arr, 0.25, method='averaged_inverted_cdf')  # 下四分位数
    print("下四分位数:", qr1)
    qr3 = np.quantile(arr, 0.75, method='averaged_inverted_cdf')  # 上四分位数
    print("上四分位数：", qr3)
    iqr = qr3 - qr1  # 计算四分位距
    print("四分位间距:", iqr)
    print("均值：", np.mean(arr))
    return iqr



def get_voxels_distribution_png(data_dir_in, label):
    p = Path(data_dir_in)
    dir_out = p.parts[-1] + "_volexs_histogram" + "_label:" + str(label)
    maybe_mkdir_p(dir_out)
    file_ctas = sorted(p.rglob("*cta1.nii.gz"))
    file_segs = sorted(p.rglob("*seg.nii.gz"))
    for file_cta, file_seg in zip(file_ctas, file_segs):
        cta_arr = sitk.GetArrayFromImage(sitk.ReadImage(file_cta))
        seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(file_seg))
        his_arr = cta_arr[seg_arr == label]
        plt.hist(his_arr, bins=50)
        plt.title(file_cta.parts[-1])
        plt.xlabel("voxel value")
        plt.ylabel("rate")
        plt.savefig(dir_out + '/' + file_cta.parts[-1] + '.png')  # 保存图片
        plt.cla()
        print("Next file>>>")

def distribution(dir_in, label_name):
    label_voxels = np.array([])
    temp = np.array([])
    p = Path(dir_in)
    files_cta = sorted(p.rglob("*cta1.nii.gz"))
    files_seg = sorted(p.rglob("*seg.nii.gz"))
    for file_cta, file_seg in zip(files_cta, files_seg):
        arr_cta = sitk.GetArrayFromImage(sitk.ReadImage(file_cta))
        arr_seg = sitk.GetArrayFromImage(sitk.ReadImage(file_seg))
        if label_name == "lumen":
            temp = arr_cta[(arr_seg == 1) | (arr_seg == 2)]
        elif label_name == "web":
            temp = arr_cta[(arr_seg == 3) | (arr_seg == 4)]
        label_voxels = np.concatenate((label_voxels, temp), axis=0)
        print("Next....")
    return label_voxels


if __name__ == '__main__':
    dir_in = "/homeb/txz/Pycharm_Project/Txz_process/raw_data"
    label_name = "web"
    ans = distribution(dir_in, label_name)
    cal_iqr(ans)
    plt.hist(ans)
    plt.show()
    print(ans.shape)
