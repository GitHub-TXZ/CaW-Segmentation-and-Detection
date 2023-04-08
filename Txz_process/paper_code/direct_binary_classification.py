import logging
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, PadD, SpatialPadd, RandFlip, RandRotate90, RandFlipd, Compose, \
    LoadImaged, RandRotate90d, Resized, ScaleIntensityd, Flipd, RandAffined, RandScaleIntensityd, AddChannel, \
    RandAffine, ScaleIntensityRanged
import glob
from reading_of_positive_sample_file_list import has_label_one
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from terminaltables import AsciiTable  # 导入AsciiTable类

FOLD = 3  # 决定进行几折交叉训练的参数，即分别为fold 1，fold 2,fold 3
MAX_EPOCH = 500  # 决定进行多少个epoch


def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_path = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task999_bifurcation_proc/imagesTr"
    seg_path = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task999_bifurcation_proc/labelsTr"
    images = sorted(glob.glob(os.path.join(data_path, "*.nii.gz")))
    # 2 binary labels for web classification: have and don't have
    d = has_label_one("/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task999_bifurcation_proc/labelsTr")
    # d = {'Doubt_01-029_1.nii.gz': True, 'Doubt_01-029_2.nii.gz': False} for example
    result = []
    for k in sorted(d.keys()):
        if d[k]:
            result.append(1)
        else:
            result.append(0)
    # result = {'Doubt_01-029_1.nii.gz': 1, 'Doubt_01-029_2.nii.gz': 0}for example
    labels = np.array(result, dtype=np.int64)
    # 定义三折划分
    kfold = KFold(n_splits=FOLD, shuffle=False, random_state=None)
    # 创建用于存储train/val文件的列表
    train_files, val_files = [], []
    # 将图像路径和对应标签组成一个字典，存储到files列表中
    files = [{"img": img, "label": label} for img, label in zip(images, labels)]
    # 将files列表进行三折划分
    for train_index, val_index in kfold.split(files):
        train_files.append([files[i] for i in train_index])
        val_files.append([files[i] for i in val_index])
    # train_files是一个含三个列表的嵌套列表
    # Define transforms for image

    # train_transforms = Compose(
    #     [AddChannel(), RandAffine(prob=0.5, translate_range=(4, 10, 10), padding_mode="border"),
    #      RandFlip(prob=0.5), RandRotate90(prob=0.5, spatial_axes=(1, 2))])

    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityRanged(keys=["img"], a_min=-10, a_max=800, b_min=0, b_max=1, clip=True),
            # ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(64, 64, 50)),
            # SpatialPadd(keys=["img"], spatial_size=(128, 128, 64), allow_missing_keys=True),
            RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 1]),
            # 后面这些是自己加的增强手段
            # RandomFlip(keys=["img"],prob=0.5),
            RandFlipd(keys=["img"], prob=0.5),
            # RandAffined(
            #     keys=["img"],
            #     prob=0.3,
            #     rotate_range=(-30.0, 30.0),
            #     translate_range=(-10.0, 10.0),
            #     scale_range=(0.9, 1.1),
            #     # shearing_range=(-5.0, 5.0),
            # ),
            # RandScaleIntensityd(keys=["img"], factors=0.1),
            # GaussianNoised(keys=["img"], mean=0.0, std=0.1),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityRanged(keys=["img"], a_min=-10, a_max=800, b_min=0, b_max=1, clip=True),
            # ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(64, 64, 50)),
            # SpatialPadd(keys=["img"], spatial_size=(128, 128, 64), allow_missing_keys=True),

        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # start a typical PyTorch training
    writer = SummaryWriter()
    Sen = [0] * FOLD
    Spec = [0] * FOLD
    Prec = [0] * FOLD
    ACC = [0] * FOLD
    AUC = [0] * FOLD
    F1 = [0] * FOLD
    Best_epoch = [0] * FOLD
    Best_metric = [0] * FOLD
    Print_FOLD = [i + 1 for i in range(FOLD)]
    for fold in range(FOLD):
        print(f"---------现在正在训练第{fold + 1}折----------")
        val_interval = 2
        best_metric = -1
        # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        # model = monai.networks.nets.resnet18(spatial_dims=3, in_channels=1, out_channels=2).to(device) #gpt生成的错误代码
        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), 1e-5)
        auc_metric = ROCAUCMetric()

        # Define dataset, data loader
        check_ds = monai.data.Dataset(data=train_files[fold], transform=train_transforms)
        check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
        check_data = monai.utils.misc.first(check_loader)
        print(check_data["img"].shape, check_data["label"])

        # create a training data loader
        train_ds = monai.data.Dataset(data=train_files[fold], transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4,
                                  pin_memory=torch.cuda.is_available())

        # create a validation data loader
        val_ds = monai.data.Dataset(data=val_files[fold], transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
        for epoch in range(MAX_EPOCH):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{MAX_EPOCH}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    for val_data in val_loader:
                        val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                        y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                        y = torch.cat([y, val_labels], dim=0)

                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                    y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                    auc_metric(y_pred_act, y_onehot)
                    auc_result = auc_metric.aggregate()
                    auc_metric.reset()

                    # 计算混淆矩阵并输出相关评价指标和参数
                    y_pred_label = y_pred.argmax(dim=1).cpu().numpy()
                    y_true_label = y.cpu().numpy()
                    cm = confusion_matrix(y_true_label, y_pred_label)
                    tn, fp, fn, tp = cm.ravel()
                    precision = precision_score(y_true_label, y_pred_label)
                    # recall = recall_score(y_true_label, y_pred_label) #等于Sen
                    specificity = tn / (tn + fp)
                    sensitivity = tp / (tp + fn)
                    f1 = f1_score(y_true_label, y_pred_label)

                    del y_pred_act, y_onehot
                    if acc_metric > best_metric:
                        best_metric = acc_metric
                        ACC[fold] = acc_metric
                        Sen[fold] = sensitivity  # 就是recall
                        Spec[fold] = specificity
                        Prec[fold] = precision
                        AUC[fold] = auc_result
                        F1[fold] = f1
                        Best_epoch[fold] = epoch + 1
                        Best_metric[fold] = best_metric

                        torch.save(model.state_dict(),
                                   "The_{}th_fold_best_metric_model_classification3d_dict.pth".format(fold + 1))
                        print("saved new best metric model")
                    print(
                        "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} precision: {:.4f} f1 score: {:.4f} Spec: {:.4f} Sen：{:.4f}".format(
                            epoch + 1, acc_metric, auc_result, precision, f1, specificity, sensitivity
                        )
                    )
                    writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
    print(f"train completed.......")
    for print_fold, best_epoch, best_acc in zip(Print_FOLD, Best_epoch, Best_metric, ):
        print("Fold: {}, best_epoch: {}, best_acc: {:.4f}".format(print_fold, best_epoch, best_acc))
    # 求平均值  # 以表格形式输出相关指标
    ACC_mean = sum(ACC) / len(ACC)
    Sen_mean = sum(Sen) / len(Sen)
    Spec_mean = sum(Spec) / len(Spec)
    Prec_mean = sum(Prec) / len(Prec)
    AUC_mean = sum(AUC) / len(AUC)
    F1_mean = sum(F1) / len(F1)

    # 创建表格数据，每一行是一个子列表
    table_data = []
    table_data.append(["Fold", "ACC", "Sen", "Spec", "Prec", "AUC", "F1"])
    row_data = [
        ["Fold-{}".format(i + 1), round(ACC[i], 2), round(Sen[i], 2), round(Spec[i], 2), round(Prec[i], 2),
         round(AUC[i], 2), round(F1[i], 2)] for i in range(len(ACC))
    ]  # 循环创建每一行数据
    table_data.extend(row_data)
    table_data.append(
        ["Mean", round(ACC_mean, 2), round(Sen_mean, 2), round(Spec_mean, 2), round(Prec_mean, 2),
         round(AUC_mean, 2),
         round(F1_mean, 2)])  # 添加平均值行

    # 创建AsciiTable对象，并设置标题和对齐方式
    table = AsciiTable(table_data)
    table.title = "Results"
    table.justify_columns[6] = 'right'  # 将最后一列右对齐
    # 打印表格
    print(table.table)
    writer.close()


if __name__ == "__main__":
    main()
