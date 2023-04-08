import numpy as np
import SimpleITK as sitk
from pathlib import Path
from medpy import metric
from Txz_process.utils import set_meta
from batchgenerators.utilities.file_and_folder_operations import *


spacing_dict = {}
p = Path("/home/txz/PSI-Seg/Txz_process/Task_ultimate_procedure/raw_cta")
files = p.rglob("*nii.gz")
for file in files:
    I = sitk.ReadImage(file)
    space = I.GetSpacing()
    spacing_dict[file.parts[-1]] = space




def prefer_metric(f0_js, f1_js, f2_js):
    dice = np.array([])
    hd_95 = np.array([])
    jc = np.array([])
    assd = np.array([])
    gt_volume = np.array([])
    pred_volume  = np.array([])
    f0_dict = load_json(f0_js)["results"]["all"]
    f1_dict = load_json(f1_js)["results"]["all"]
    f2_dict = load_json(f2_js)["results"]["all"]
    for i in f0_dict:
        pat_id = i["test"].split('/')[-1]
        dice = np.append(dice, i["1"]["Dice"])
        hd_95 = np.append(hd_95, i["1"]["Hausdorff Distance 95"])
        jc = np.append(jc, i["1"]["Jaccard"])
        assd = np.append(assd, i["1"]["Avg. Symmetric Surface Distance"])
        gt_volume  = np.append(gt_volume, i["1"]["Total Positives Reference"]*np.prod(spacing_dict[pat_id])/1000)
        pred_volume  = np.append(pred_volume, i["1"]["Total Positives Test"]*np.prod(spacing_dict[pat_id])/1000)

    for i in f1_dict:
        pat_id = i["test"].split('/')[-1]
        dice = np.append(dice, i["1"]["Dice"])
        hd_95 = np.append(hd_95, i["1"]["Hausdorff Distance 95"])
        jc = np.append(jc, i["1"]["Jaccard"])
        assd = np.append(assd, i["1"]["Avg. Symmetric Surface Distance"])
        gt_volume = np.append(gt_volume, i["1"]["Total Positives Reference"] * np.prod(spacing_dict[pat_id]) / 1000)
        pred_volume = np.append(pred_volume, i["1"]["Total Positives Test"] * np.prod(spacing_dict[pat_id]) / 1000)

    for i in f2_dict:
        pat_id = i["test"].split('/')[-1]
        dice = np.append(dice, i["1"]["Dice"])
        hd_95 = np.append(hd_95, i["1"]["Hausdorff Distance 95"])
        jc = np.append(jc, i["1"]["Jaccard"])
        assd = np.append(assd, i["1"]["Avg. Symmetric Surface Distance"])
        gt_volume = np.append(gt_volume, i["1"]["Total Positives Reference"] * np.prod(spacing_dict[pat_id]) / 1000)
        pred_volume = np.append(pred_volume, i["1"]["Total Positives Test"] * np.prod(spacing_dict[pat_id]) / 1000)

    mean_dice = np.mean(dice)
    std_dice = np.std(dice)
    mean_hd_95 = np.mean(hd_95)
    std_hd_95 = np.std(hd_95)
    corrcoef = np.corrcoef(gt_volume, pred_volume)[0][1]
    print(f"dice:{mean_dice}±{std_dice},hd_95:{mean_hd_95}±{std_hd_95},corrcoef:{corrcoef}")


if __name__ == '__main__':
    f0_js = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task666_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw/summary.json"
    f1_js = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task666_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/summary.json"
    f2_js = "/home/txz/PSI-Seg/nnUNet_trained_models/nnUNet/3d_fullres/Task666_utimate_proc/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/validation_raw/summary.json"
    prefer_metric(f0_js, f1_js, f2_js)

