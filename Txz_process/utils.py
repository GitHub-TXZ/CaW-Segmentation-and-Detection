# -*- coding:utf-8 -*-
import json
import pandas as pd
import os


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def set_meta(itk, origin, spacing, direction):
    itk.SetDirection(direction)
    itk.SetOrigin(origin)
    itk.SetSpacing(spacing)


def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i + 1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i + 1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


# 获取一个目录下的子目录
def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def json_to_csv(json_path):
    with open(json_path) as f:
        f_data = json.load(f)
    pat_id = []
    asd = []
    dice = []
    hd = []
    jc = []
    vol_gt = []
    vol_pred = []
    for k, v in f_data.items():
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
    dict = {"pat_id": pat_id, "asd": asd, "dice": dice, "hd": hd, "jc": jc, "vol_gt": vol_gt,
            "vol_pred": vol_pred}
    df = pd.DataFrame(dict)
    # 保存 dataframe
    df.to_csv(json_path.split('.')[-2]+'.csv')



if __name__ == '__main__':
    file_path = "Task004Task005Task006_Merged_mertrics_info.json"
    json_to_csv(file_path)
