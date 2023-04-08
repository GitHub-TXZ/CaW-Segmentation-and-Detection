# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
import math


# 函数：计算相关系数
def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
    corr_factor = cov_ab / sq
    return corr_factor


filename = '/homeb/txz/Pycharm_Project/Txz_process/TaskslidingWindow015/carotid_lumen_merge_3flod_metric.csv'
dice = []
hd = []
vol_gt = []
vol_pred = []
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到data中
        if row[3] != "NA":
            dice.append(float(row[3]))  # 选择某一列加入到data数组中
            hd.append(float(row[4]))
            vol_gt.append(float(row[6]))
            vol_pred.append(float(row[7]))

b_s = pd.Series(vol_gt)
a_s = pd.Series(vol_pred)
cor1 = a_s.corr(b_s)
cor2 = calc_corr(vol_gt, vol_pred)
print(cor1, cor2)
print(np.mean(dice), np.std(dice), np.mean(hd), np.std(hd), np.corrcoef(vol_gt, vol_pred))
