# -*- coding:utf-8 -*-
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
# src_cta = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/carotid_lumen_patch"
# src_seg = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/GT_seg_maybe_include_web"
# dst_cta = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Separate_for_test_cta"
# dst_seg = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Separate _for_test_seg"
# have_web_path = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/have_web.json"
# have_web = load_json(have_web_path)
# all_cta = os.listdir(
#     "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/carotid_lumen_patch")
# all_seg = os.listdir(
#     "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task015_sliding_for_web/GT_seg_maybe_include_web")
#
# for i, j in zip(all_cta, all_seg):
#     if i[:-12] + ".nii.gz" not in have_web:
#         shutil.copy(os.path.join(src_cta, i), os.path.join(dst_cta, i))
#         shutil.copy(os.path.join(src_seg, j), os.path.join(dst_seg, j))

# 按照条件删除文件夹下的指定文件，删除操作前一定要记得先备份
# import os
#
# with open("/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/ten_have_web.pkl",
#           "rb") as f:
#     ten_have_web = pickle.load(f)
# rootdir_cta = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Separate_for_test_cta"
# rootdir_seg = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/Separate _for_test_seg"
# filelist_cta = os.listdir(rootdir_cta)
# filelist_seg = os.listdir(rootdir_seg)

# for file in filelist_cta:
#     if file[:-12] + ".nii.gz" not in ten_have_web:
#         del_file = rootdir_cta + '/' + file
#         os.remove(del_file)
#         print("removing:", del_file)
# for file in filelist_seg:
#     if file not in ten_have_web:
#         del_file = rootdir_seg + '/' + file
#         os.remove(del_file)
#         print("removing:", del_file)

# 进行二分类web检测任务
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import pickle

GT_seg_path = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/test"
Pred_seg_path = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/prediction"
GT_files = sorted(Path(GT_seg_path).rglob("*.nii.gz"))
Pred_files = sorted(Path(Pred_seg_path).rglob("*.nii.gz"))

with open("/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/30_have_web.pkl",
          "rb") as f:
    ten_have_web = pickle.load(f)
fpr = []
tpr = []
thresholds = [i for i in range(10, 60, 10)]
pred_have_web = [[] for i in range(len(thresholds))]
pred_not_have_web = [[] for i in range(len(thresholds))]
fp_more_than_50 = []
for Pred_file in Pred_files:
    pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(Pred_file))
    for i, th in enumerate(thresholds):
        if np.sum(pred_arr, axis=(1, 2)).sum() >= th:
            pred_have_web[i].append(Pred_file.parts[-1])
        else:
            pred_not_have_web[i].append(Pred_file.parts[-1])
for i, th in enumerate(thresholds):
    confusion_matrix = np.zeros((2, 2))
    for j in pred_not_have_web[i]:
        if j in ten_have_web:
            confusion_matrix[1][0] += 1
        else:
            confusion_matrix[0][0] += 1
    for k in pred_have_web[i]:
        if k in ten_have_web:
            confusion_matrix[1][1] += 1
        else:
            confusion_matrix[0][1] += 1
            # 统计49个中有哪些是>50的
            if th == 50:
                fp_more_than_50.append(k)

    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    ACC = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    fpr.append(FP / (FP + TN))
    tpr.append(TP / (TP + FN))
    print(confusion_matrix, "ACC:%f,Sensitivity:%f,Specificity:%f" % (ACC, Sensitivity, Specificity))
    print(len(fp_more_than_50),fp_more_than_50)


auc = auc(fpr, tpr)
print("auc:", auc)
plt.plot(fpr, tpr, color="r", linestyle="--", marker="*", linewidth=1.0)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("Receiver Operating Characteristic")
plt.show()
