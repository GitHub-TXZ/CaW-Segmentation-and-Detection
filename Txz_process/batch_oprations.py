# -*- coding:utf-8 -*-
base = '/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/imagesTr/case_00000_0000.nii.gz'
# 写文件
f_out = gzip.open(base, "r")

import nibabel as nb
import os
base = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/imagesTr"
for file in os.listdir(base):
        nii_img = nb.load(os.path.join(base, file))
        nii_data = nii_img.get_fdata()
        new_data = nii_data.copy()
        # 省略一些处理data的骚操作,比如：
        # new_data[new_data>4] = 0
        # 把仿射矩阵和头文件都存下来
        affine = nii_img.affine.copy()
        hdr = nii_img.header.copy()
        # 形成新的nii文件
        new_nii = nb.Nifti1Image(new_data, affine, hdr)
        # 保存nii文件，后面的参数是保存的文件名
        nb.save(new_nii, os.path.join(base, file))
        print(file)


## using simpleITK to load and save data.
# import SimpleITK as sitk
# import os
base = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/imagesTr"
for file in os.listdir(base):
        itk_img = sitk.ReadImage(os.path.join(base,file))
        img = sitk.GetArrayFromImage(itk_img)
        print("img shape:",img.shape)
        ## save
        out = sitk.GetImageFromArray(img)
        sitk.WriteImage(out, os.path.join(base, file))

import SimpleITK as sitk
import nibabel as nb
import os
base1 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/imagesTr"
base2 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/labelsTr"
for file1 in os.listdir(base1):
       for file2 in os.listdir(base2):
            if file1.split('.')[0].split('_')[-2] == file2.split('.')[0].split('_')[-1]:
               nii_img = nb.load(os.path.join(base1, file1))
               nii_cankao = nb.load(os.path.join(base2, file2))
               nii_data = nii_img.get_fdata()
               new_data = nii_data.copy()
               # 省略一些处理data的骚操作,比如：
               # new_data[new_data>4] = 0
               # 把仿射矩阵和头文件都存下来
               affine = nii_cankao.affine.copy()
               hdr = nii_cankao.header.copy()
               # 形成新的nii文件
               new_nii = nb.Nifti1Image(new_data, affine, hdr)
               # 保存nii文件，后面的参数是保存的文件名
               nb.save(new_nii, os.path.join(base1, file1))
               print(file1)


# 批量修改数据集
import SimpleITK as sitk
import nibabel as nb
import numpy as np
import os
base1 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/imagesTr"
base2 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task066_carotid/labelsTr"
base3 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task077_carotid/imagesTr"
base4 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task077_carotid/labelsTr"
for file in os.listdir(base2):
   I = sitk.ReadImage(os.path.join(base2,file))
   origin = I.GetOrigin()
   spacing = I.GetSpacing()
   direction = I.GetDirection()
   img = sitk.GetArrayFromImage(I)
   img[img > 2] = 0
   img[img > 1] = 1
   zrange =  np.sum(img, axis=(1, 2))
   tempz = np.where(zrange != 0)
   zmin = tempz[0][0]
   zmax = tempz[0][-1]

   xrange = np.sum(img, axis=(0, 2))
   tempx = np.where(xrange != 0)
   xmin = tempx[0][0]
   xmax = tempx[0][-1]

   yrange = np.sum(img, axis=(0, 1))
   tempy = np.where(yrange != 0)
   ymin = tempy[0][0]
   ymax = tempy[0][-1]

   patch = img[(zmin-10):(zmax+10),(xmin-10):(xmax+10),(ymin-10):(ymax+10)]
   out = sitk.GetImageFromArray(patch)
   sitk.WriteImage(out, os.path.join(base4,file))

   I2 = sitk.ReadImage(os.path.join(base4, file))
   I2.SetDirection(direction)
   I2.SetOrigin(origin)
   I2.SetSpacing(spacing)
   img2 = sitk.GetArrayFromImage(I2)
   out2 =  sitk.GetImageFromArray(img2)
   sitk.WriteImage(out2, os.path.join(base4, file))

   for file2 in os.listdir(base1):
      if file2.split('.')[0].split('_')[-2] == file.split('.')[0].split('_')[-1]:
            I3 = sitk.ReadImage(os.path.join(base1,file2))
            img3 = sitk.GetArrayFromImage(I3)
            patch2 = img3[(zmin-10):(zmax+10),(xmin-10):(xmax+10),(ymin-10):(ymax+10)]
            out3 = sitk.GetImageFromArray(patch2)
            sitk.WriteImage(out3, os.path.join(base3,file2))

            I4 = sitk.ReadImage(os.path.join(base3, file2))
            I4.SetDirection(direction)
            I4.SetOrigin(origin)
            I4.SetSpacing(spacing)
            img4 = sitk.GetArrayFromImage(I4)
            out4 = sitk.GetImageFromArray(img4)
            sitk.WriteImage(out4, os.path.join(base3, file2))
print("*************************************************************")
def set_meta(itk,*args):
    itk.SetDirection(args[0][0])
    itk.SetOrigin(args[0][1])
    itk.SetSpacing(args[0][2])

def get_meta(itk):
    return itk.GetDirection(),itk.GetOrigin(),itk.GetSpacing()
import numpy as np
import os


def set_meta(itk, origin, spacing, direction):
    itk.SetDirection(direction)
    itk.SetOrigin(origin)
    itk.SetSpacing(spacing)


### 预测前处理函数
import SimpleITK as sitk
from pathlib import Path


def pred_preprocess(file_in, file_out):
    base = file_in
    out = file_out
    p = Path(base)
    files = p.rglob("*case_00055/imaging.nii.gz")
    for file in files:
        itk = sitk.ReadImage(file)
        d = itk.GetDirection()
        o = itk.GetOrigin()
        s = itk.GetSpacing()
        arr = sitk.GetArrayFromImage(itk)
        # arr = arr.squeeze()
        # 236+-30，255+-50，188+-50 and 321+-50
        print(arr.shape)
        new_arr1 = arr[205:247, 176:224, 300:352]
        new_arr2 = arr[1:3, 1:3, 1:3]
        itk1 = sitk.GetImageFromArray(new_arr1)
        print(itk1.GetSize())
        set_meta(itk1, o, s, d)
        itk2 = sitk.GetImageFromArray(new_arr2)
        set_meta(itk2, o, s, d)
        sitk.WriteImage(itk1, out + "/1_0000.nii.gz")
        sitk.WriteImage(itk2, out + "/2_0000.nii.gz")
        return arr.shape


def post_process(raw_shape, im1, im2):
    base = '/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/prediction'
    out = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/prediction"
    p = Path(base)
    # files = p.glob("*.nii.gz")
    # file = [i for i in files]
    arr1 = sitk.GetArrayFromImage(sitk.ReadImage(im1))
    d = sitk.ReadImage(im1).GetDirection()
    o = sitk.ReadImage(im1).GetOrigin()
    s = sitk.ReadImage(im1).GetSpacing()
    arr2 = sitk.GetArrayFromImage(sitk.ReadImage(im2))
    new_arr = np.zeros(raw_shape)
    new_arr[205:247, 176:224, 300:352] = arr1
    new_arr[1:3, 1:3, 1:3] = arr2
    new_itk = sitk.GetImageFromArray(new_arr)
    set_meta(new_itk, o, s, d)
    sitk.WriteImage(new_itk, out + "/1_2.nii.gz")


def modify_seg_web(file_in, file_out):
    arr_itk = sitk.ReadImage(file_in)
    arr = sitk.GetArrayFromImage(arr_itk)
    # arr [ arr < 3] = 0
    # arr [ arr > 4 ] = 0
    # arr[ arr == 3] = 1
    # arr[ arr == 4] = 1
    arr[239:242, 269:271, 171:175] = 1
    new_itk = sitk.GetImageFromArray(arr)
    new_itk.CopyInformation(arr_itk)
    sitk.WriteImage(new_itk, file_out)


if __name__ == "__main__":
    file_in = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/temporary"
    file_out = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/imagesTs"
    pred_preprocess(file_in, file_out)
    os.system("export nnUNet_raw_data_base=""/homeb/txz/Pycharm_Project/nnUNet_raw_data_base""")
    os.system("export nnUNet_preprocessed=""/homeb/txz/Pycharm_Project/nnUNet_preprocessed""")
    os.system("export RESULTS_FOLDER=""/homeb/txz/Pycharm_Project/nnUNet_trained_models""" )
    # os.system("nnUNet_predict -i /homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/imagesTs -o /homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/prediction -t 99 -m 3d_fullres")
    im1 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/prediction/1.nii.gz"
    im2 = "/homeb/txz/Pycharm_Project/nnUNet_raw_data_base/nnUNet_raw_data/Task099_carotid/prediction/2.nii.gz"
    raw_shape = (588,512,512)
    print(type(raw_shape))
    post_process(raw_shape,im1, im2)
    modify_seg_web(r'D:\CTA_Analyze\pytorch-UNet\case_00055\1_2.nii.gz',
                   r'D:\CTA_Analyze\pytorch-UNet\case_00055\1_2.nii.gz_modify.nii.gz')
