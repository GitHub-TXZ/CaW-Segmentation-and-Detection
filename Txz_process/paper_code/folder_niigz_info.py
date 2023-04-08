import SimpleITK as sitk
import os

data_dir = "/home/txz/PSI-Seg/nnUNet_raw_data_base/nnUNet_raw_data/Task999_bifurcation_proc/imagesTr"

# iterate through all files in the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".nii.gz"):
        filepath = os.path.join(data_dir, filename)

        # read the image file
        image = sitk.ReadImage(filepath)

        # get image size, pixel spacing and number of components
        size = image.GetSize()
        spacing = image.GetSpacing()
        components = image.GetNumberOfComponentsPerPixel()

        # print information about the image
        print(f"File: {filename}")
        print(f"Image size: {size}")
        print(f"Pixel spacing: {spacing}")
        print(f"Number of components: {components}")
        print()
