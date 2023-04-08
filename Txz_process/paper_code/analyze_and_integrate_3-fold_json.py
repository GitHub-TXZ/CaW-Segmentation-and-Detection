import math

from numpy import nan
from sklearn.utils import resample

new_results = {}
json1 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task912_脖子劈成两半直接分割CaW/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json"
json2 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task912_脖子劈成两半直接分割CaW/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json"
json3 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task912_脖子劈成两半直接分割CaW/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json"
# #
# json1 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task346_web_bbox_remain_osd/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json"
# json2 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task346_web_bbox_remain_osd/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json"
# json3 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task346_web_bbox_remain_osd/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json"

# json1 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task999_bifurcation_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json"
# json2 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task999_bifurcation_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json"
# json3 = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task999_bifurcation_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json"
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def dice_hd(dice_list, hd_list):
    dice_mean = np.mean(dice_list)
    dice_std = np.std(dice_list)
    hd_mean = np.mean(hd_list)
    hd_std = np.std(hd_list)
    print("参与计算的dice的个数：", len(dice_list))
    print("参与计算hd的个数：", len(hd_list))
    print(f"Dice Coefficient Mean: {dice_mean:.4f}, Standard Deviation: {dice_std:.4f}")
    print(f"Hausdorff Distance Mean: {hd_mean:.2f}, Standard Deviation: {hd_std:.2f}")


a = load_json(json1)["results"]["all"]
b = load_json(json2)["results"]["all"]
c = load_json(json3)["results"]["all"]
a.extend(b)
a.extend(c)
vol_gt_list = []
vol_pred_list = []
dice_list = []
hd_list = []
results = a
tn = fp = fn = tp = 0
TH = 70
# 通过循坏找到一个好的阈值，80
for idx, ele in enumerate(results):
    # print(ele)
    ID = ele["reference"].split('/')[-1]
    new_results[ID] = {"Dice": ele["1"]["Dice"], "HD": ele["1"]["Hausdorff Distance 95"],
                       "Total Positives Reference": ele["1"]["Total Positives Reference"],
                       "Total Positives Test": ele["1"]["Total Positives Test"], "reference": ele["reference"],
                       "test": ele["test"]}
# new_results中保存了所有需要分析的病人的详细信息
# 计算体积
for key, value in new_results.items():
    # 对于本来就有CaW的y样本
    if value["Total Positives Reference"] != 0:
        vol_gt = value["Total Positives Reference"] * (np.prod(sitk.ReadImage(value["reference"]).GetSpacing()))
        vol_pred = value["Total Positives Test"] * (np.prod(sitk.ReadImage(value["test"]).GetSpacing()))
        vol_gt_list.append(vol_gt)
        vol_pred_list.append(vol_pred)

        if value["Dice"] != 1:
            dice_list.append(value["Dice"])
        else:
            print("Debug: Dice value is 1 and was not added to the list.")

        if not math.isnan(value["HD"]):
            hd_list.append(value["HD"])
        else:
            print("Value is NaN")

        if value["Total Positives Test"] >= TH:
            tp += 1
        else:
            fn += 1
    # 对于本来就没有CaW的样本
    else:
        if value["Total Positives Test"] >= TH:
            fp += 1
        else:
            tn += 1

dice_hd(dice_list, hd_list)
print("TH = {}, tn = {}, fp = {}, fn = {}, tp = {}".format(TH, tn, fp, fn, tp))
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
sensitivity = recall = tp / (tp + fn)
specificity = tn / (tn + fp)
f1_score = 2 * precision * recall / (precision + recall)
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
auc = (tpr - fpr + 1.0) / 2

print(
    f"Accuracy: {accuracy} Precision: {precision} Sensitivity: {sensitivity} Specificity: {specificity} AUC: {auc} F1 Score: {f1_score}\n\n")


def picture(vol_gt_list, vol_pred_list):
    # Pearson correlation plot
    # Pearson correlation plot
    corr, _ = pearsonr(vol_gt_list, vol_pred_list)
    plt.scatter(vol_gt_list, vol_pred_list, s=50, c='gray')
    plt.title(f"Pearson Correlation (corr={corr:.2f})")
    plt.xlabel("Ground Truth Volume(mm³)")
    plt.ylabel("Predicted Volume(mm³)")
    z = np.polyfit(vol_gt_list, vol_pred_list, 1)
    p = np.poly1d(z)
    plt.plot(vol_gt_list, p(vol_gt_list), "r--")
    plt.rcParams['axes.facecolor'] = 'none'
    plt.show()

    # Bland-Altman plot
    diff = np.array(vol_gt_list) - np.array(vol_pred_list)
    mean = np.mean([vol_gt_list, vol_pred_list], axis=0)
    plt.scatter(mean, diff, s=50, c='gray')

    ci = 1.96 * np.std(diff) / np.sqrt(len(diff))
    plt.axhline(y=np.mean(diff), color='gray', linestyle='--')
    plt.axhline(y=np.mean(diff) + ci, color='gray', linestyle='--')

    plt.axhline(y=np.mean(diff) - ci, color='gray', linestyle='--')

    plt.axhline(y=np.mean(diff) + 1.96 * np.std(diff), color='r', linestyle='--')
    # plt.text(0, np.mean(diff) + 1.96 * np.std(diff), f"{np.mean(diff) + 1.96 * np.std(diff):.2f}", ha='right',
    #          va='bottom')
    plt.axhline(y=np.mean(diff) - 1.96 * np.std(diff), color='r', linestyle='--')

    plt.title("Bland-Altman Plot")
    plt.xlabel("Mean Volume(mm³)")
    plt.ylabel("Difference (Ground Truth - Predicted)(mm³)")
    plt.rcParams['axes.facecolor'] = 'none'
    plt.show()


# df = pd.DataFrame({'Ground Truth Volume': vol_gt_list, 'Predicted Volume': vol_pred_list})
# df.to_excel('volume_predictions_GT.xlsx', index=False)
# picture(vol_gt_list, vol_pred_list)
pearson = np.corrcoef(vol_gt_list, vol_pred_list)
print("pearson:", pearson)
# dice_hd(dice_list, hd_list)

# 使用bootstrap进行自助抽样，计算置信区间
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score


# Define function to calculate bootstrap confidence intervals
def bootstrap_ci(data, func, alpha=0.05, n_bootstraps=1000):
    """Calculate bootstrap confidence intervals for a given function"""
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # Resample data with replacement
        resampled_data = resample(data, replace=True, n_samples=len(data))
        # Calculate function on resampled data
        score = func(resampled_data)
        bootstrapped_scores.append(score)
    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrapped_scores, alpha / 2 * 100)
    upper_bound = np.percentile(bootstrapped_scores, (1 - alpha / 2) * 100)
    return (lower_bound, upper_bound)


# Define function to calculate binary classification metrics
def binary_metrics(y_true, y_pred):
    """Calculate binary classification metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred)
    spec = recall_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return (acc, prec, sen, spec, auc, f1)


#

# # Define data 对混淆矩阵进行解析编码
def confusion_to_metric(tn, fp, fn, tp):
    y_true = [0] * (tn + fp) + [1] * (fn + tp)
    y_pred = [0] * tn + [1] * fp + [0] * fn + [1] * tp
    metrics = binary_metrics(y_true, y_pred)
    # Calculate binary classification metrics with bootstrap confidence intervals
    data = np.array(list(zip(y_true, y_pred)))
    acc_ci = bootstrap_ci(data, lambda x: binary_metrics(x[:, 0], x[:, 1])[0])
    prec_ci = bootstrap_ci(data, lambda x: binary_metrics(x[:, 0], x[:, 1])[1])
    sen_ci = bootstrap_ci(data, lambda x: binary_metrics(x[:, 0], x[:, 1])[2])
    spec_ci = bootstrap_ci(data, lambda x: binary_metrics(x[:, 0], x[:, 1])[3])
    auc_ci = bootstrap_ci(data, lambda x: binary_metrics(x[:, 0], x[:, 1])[4])
    f1_ci = bootstrap_ci(data, lambda x: binary_metrics(x[:, 0], x[:, 1])[5])
    print(f"Accuracy: {metrics[0]:.4f}-({acc_ci[0]:.4f}, {acc_ci[1]:.4f})")
    print(f"Precision: {metrics[1]:.4f}-({prec_ci[0]:.4f}, {prec_ci[1]:.4f})")
    print(f"Sensitivity: {metrics[2]:.4f}-({sen_ci[0]:.4f}, {sen_ci[1]:.4f})")
    print(f"Specificity: {metrics[3]:.4f}-({spec_ci[0]:.4f}, {spec_ci[1]:.4f})")
    print(f"AUC: {metrics[4]:.4f}-({auc_ci[0]:.4f}, {auc_ci[1]:.4f})")
    print(f"F1 Score: {metrics[5]:.4f}-({f1_ci[0]:.4f}, {f1_ci[1]:.4f})")


if __name__ == "__main__":
    # confusion_to_metric(44, 11, 6, 55)
    confusion_to_metric(44, 11, 12, 49)
    # confusion_to_metric(52, 3, 6, 55)
