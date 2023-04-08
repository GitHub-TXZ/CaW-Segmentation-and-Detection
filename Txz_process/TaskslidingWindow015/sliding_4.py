# -*- coding: utf-8 -*-
from __future__ import absolute_import
from Txz_process.utils import *
import pathlib
import shutil
import os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from collections import OrderedDict
import numpy as np
import pickle
import SimpleITK as sitk
import numpy as np
import os
import SimpleITK as sitk
from medpy import metric
from Txz_process.utils import save_json
from pathlib import Path
import collections
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# 将三折的结果融合进一个csv文件中
def metric_json_merge(json1, json2, json3, have_web1 = None, have_web2 = None, have_web3 = None):
    if have_web1 != None:
        w1 = load_pickle(have_web1)
        w2 = load_pickle((have_web2))
        w3 = load_pickle((have_web3))
        w = [w1, w2, w3]

    pat_id = []
    asd = []
    dice = []
    hd = []
    jc = []
    vol_gt = []
    vol_pred = []

    j1 = load_json(json1)
    j2 = load_json(json2)
    j3 = load_json(json3)
    j = [j1, j2, j3]

    for i in range(3):
        for k, v in j[i].items():
            if have_web1 != None:
                if k != "mean_dice" and k in w[i]:
                    pat_id.append(k)
                    if v != "NA":
                        asd.append(v["asd"])
                        dice.append(v["dice"])
                        hd.append(v["hd"])
                        jc.append(v["jc"])
                        vol_gt.append(v["vol_gt"])
                        vol_pred.append(v["vol_pred"])
                    else:
                        asd.append("NA")
                        dice.append("NA")
                        hd.append("NA")
                        jc.append("NA")
                        vol_gt.append("NA")
                        vol_pred.append("NA")
            else:
                if k != "mean_dice":
                    pat_id.append(k)
                    if v != "NA":
                        asd.append(v["asd"])
                        dice.append(v["dice"])
                        hd.append(v["hd"])
                        jc.append(v["jc"])
                        vol_gt.append(v["vol_gt"])
                        vol_pred.append(v["vol_pred"])
                    else:
                        asd.append("NA")
                        dice.append("NA")
                        hd.append("NA")
                        jc.append("NA")
                        vol_gt.append("NA")
                        vol_pred.append("NA")

    dice_mean = np.mean([i for i in dice if i != "NA"])
    dice_std = np.std([i for i in dice if i != "NA"], ddof=1)
    hd_mean = np.mean([i for i in hd if i != "NA"])
    hd_std = np.std([i for i in hd if i != "NA"], ddof=1)
    pccs = np.corrcoef([i for i in vol_pred if i != "NA"], [i for i in vol_gt if i != "NA"])
    dict = {"pat_id": pat_id, "asd": asd, "dice": dice, "hd": hd, "jc": jc, "vol_gt": vol_gt,
            "vol_pred": vol_pred, "dice_mean": [dice_mean]*len(dice), "dice_std": [dice_std]*len(dice), "hd_mean": [hd_mean]*len(dice), "hd_std": [hd_std]*len(dice),
            "Pearson correlation coefficient between vol_gt and vol_pred": [pccs]*len(dice)}
    df = pd.DataFrame(dict)
    # 保存 dataframe
    df.to_csv('merge_3flod_metric.csv')


json1 = "/homeb/txz/Pycharm_Project/Txz_process/lumen1.json"
json2 = "/homeb/txz/Pycharm_Project/Txz_process/lumen2.json"
json3 = "/homeb/txz/Pycharm_Project/Txz_process/lumen3.json"
# have_web1 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task016_sliding_for_web_V2/ten_have_web.pkl"
# have_web2 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task018_fuzhu/27_have_web.pkl"
# have_web3 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task019_fuzhu/30_have_web.pkl"
# metric_json_merge(json1, json2, json3, have_web1, have_web2, have_web3)
metric_json_merge(json1, json2, json3)


