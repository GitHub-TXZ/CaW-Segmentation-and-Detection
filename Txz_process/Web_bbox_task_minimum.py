import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
from skimage import measure
from skimage.morphology import binary_dilation, binary_erosion, cube
from scipy import ndimage
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *


def generate_json(path_tr, path_label, nnunet_dir):
    # 生成nnunet训练和测试所需要的json文件
    p_mydata_cta = Path(path_tr)
    p_mydata_seg = Path(path_label)
    join = os.path.join
    files_cta1 = sorted(p_mydata_cta.rglob("*.nii.gz"))
    files_seg = sorted(p_mydata_seg.rglob("*.nii.gz"))
    json_dict = {}
    json_dict['name'] = "Tan Xian Zhen"
    json_dict['description'] = "slide web segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "TXZ"
    json_dict['licence'] = "TXZ"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CTA",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "web",
    }
    json_dict['numTraining'] = len(files_cta1)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': join("imagesTr/", i.parts[-1][:-12] + ".nii.gz"),
                              "label": join("labelsTr/", j.parts[-1][:-7] + ".nii.gz")} for
                             i, j in
                             zip(files_cta1, files_seg)]
    json_dict['test'] = []
    save_json(json_dict, join(nnunet_dir, "dataset.json"))


def print_info(path):
    import SimpleITK as sitk
    image = sitk.ReadImage(path)
    print('Origin:', image.GetOrigin())
    print('Spacing:', image.GetSpacing())
    print('Direction:', image.GetDirection())


def get_bbox(mask):
    """
    Get bounding box of non-zero elements in a 3D mask
    """
    mask = mask.astype(np.bool)
    bbox = np.zeros((3, 2), dtype=np.int32)
    for i in range(3):
        mask_any = np.any(np.any(mask, axis=0), axis=0)
        if not np.any(mask_any):
            bbox[i] = [0, mask.shape[i]]
        else:
            bbox[i] = [np.argmax(mask_any), mask.shape[i] - np.argmax(np.flip(mask_any))]
        mask = np.swapaxes(mask, 0, 1)
    return bbox


def crop_volume(volume, bbox):
    """
    Crop a 3D volume using a bounding box
    """
    return volume[bbox[0, 0]:bbox[0, 1], bbox[1, 0]:bbox[1, 1], bbox[2, 0]:bbox[2, 1]]


def get_largest_cc(mask):
    """
    Get largest connected component of a 3D mask
    """
    labeled_mask, num_labels = label(mask)
    if num_labels == 0:
        return mask
    largest_cc = max(regionprops(labeled_mask), key=lambda x: x.area).label
    return labeled_mask == largest_cc


count = 0

def copy_Information(srcImage,dstImage):
    dstImage.SetOrigin(srcImage.GetOrigin())
    dstImage.SetDirection(srcImage.GetDirection())
    dstImage.SetSpacing(srcImage.GetSpacing())


def process_subject(subject_dir, output_dir):
    """
    Process a single subject directory
    """
    # 我的文件并不是在subject_dir下面，这个下面有两级目录
    global count
    image_path = os.path.join(base, subject_dir,
                              [f for f in os.listdir(os.path.join(base, subject_dir)) if
                               f.endswith("cta1.nii.gz")][0])
    mask_path = os.path.join(base, subject_dir,
                             [f for f in os.listdir(os.path.join(base, subject_dir)) if
                              f.endswith("_seg.nii.gz")][0])
    raw_itk_image = sitk.ReadImage(image_path)
    raw_itk_mask = sitk.ReadImage(mask_path)
    image = sitk.GetArrayFromImage(raw_itk_image)
    mask = sitk.GetArrayFromImage(raw_itk_mask)
    # 将mask中标签值为3的区域标记为1，其余区域标记为0
    labeled_mask = (mask == 3).astype(int)
    # 使用measure.label函数对标记后的mask进行标记
    labeled_mask, num_labels = ndimage.label(labeled_mask)
    if num_labels != 0:
        # 使用measure.regionprops函数获取标记区域的属性
        regions = measure.regionprops(labeled_mask)
        # 遍历标记区域，获取标签值为3的边界框
        for region in regions:
            if region.label == 1:
                bbox = region.bbox
                bbox = np.array(bbox).reshape(2, 3).T
                break
        print("label3:", bbox)
        image_cropped = crop_volume(image, bbox)
        mask_cropped = crop_volume(mask, bbox)
        mask_cropped[mask_cropped != 3] = 0
        mask_cropped[mask_cropped == 3] = 1
        image_cropped = image_cropped.astype(np.float32)
        mask_cropped = mask_cropped.astype(np.uint8)
        image_nifti = sitk.GetImageFromArray(image_cropped)
        # image_nifti.CopyInformation(raw_itk_image)
        copy_Information(raw_itk_image,image_nifti)
        mask_nifti = sitk.GetImageFromArray(mask_cropped)
        # mask_nifti.CopyInformation(raw_itk_mask)
        copy_Information(raw_itk_mask, mask_nifti)
        sitk.WriteImage(image_nifti, os.path.join(output_dir, "imagesTr", "image%02d_0000.nii.gz" % count))
        sitk.WriteImage(mask_nifti, os.path.join(output_dir, "labelsTr", "mask%02d.nii.gz" % count))
        count += 1

    labeled_mask = (mask == 4).astype(int)
    # 使用measure.label函数对标记后的mask进行标记
    labeled_mask, num_labels = ndimage.label(labeled_mask)
    if num_labels != 0:
        # 使用measure.regionprops函数获取标记区域的属性
        regions = measure.regionprops(labeled_mask)
        # 遍历标记区域，获取标签值为3的边界框
        for region in regions:
            if region.label == 1:
                bbox = region.bbox
                bbox = np.array(bbox).reshape(2, 3).T
                break
        print("label4:", bbox)
        image_cropped = crop_volume(image, bbox)
        mask_cropped = crop_volume(mask, bbox)
        mask_cropped[mask_cropped != 4] = 0
        mask_cropped[mask_cropped == 4] = 1
        image_cropped = image_cropped.astype(np.float32)
        mask_cropped = mask_cropped.astype(np.uint8)
        image_nifti = sitk.GetImageFromArray(image_cropped)
        # image_nifti.CopyInformation(raw_itk_image)
        copy_Information(raw_itk_image, image_nifti)
        mask_nifti = sitk.GetImageFromArray(mask_cropped)
        # mask_nifti.CopyInformation(raw_itk_mask)
        copy_Information(raw_itk_mask, mask_nifti)
        sitk.WriteImage(image_nifti, os.path.join(output_dir, "imagesTr", "image%02d_0000.nii.gz" % count))
        sitk.WriteImage(mask_nifti, os.path.join(output_dir, "labelsTr", "mask%02d.nii.gz" % count))
        count += 1

    # bbox = get_bbox(mask)

def batch_rename(path):
    import os
    folder_path = path
    for filename in os.listdir(folder_path):
        if "mask" in filename:
            new_filename = filename.replace("mask", "image")
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

base = "/home/txz/PSI-Seg/Txz_process/raw_data"
task_path = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task346_web_bbox_remain_osd"
# for folder in os.listdir(base):
#     process_subject(folder, task_path)
path = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task346_web_bbox_remain_osd/labelsTr"
batch_rename(path) #重命名之后再产生json文件
generate_json(task_path + '/imagesTr', task_path + "/labelsTr", task_path)
