
import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt  # plt 用于显示图片


def save_array_as_nii_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if reference_name is not None:
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


image_path = r'G:\csvd\segunet\epvs2930/'
image_arr = glob.glob(str(image_path) + str("/*"))
image_arr.sort()
for i in range(2):
    allImg = []
    allImg = np.zeros([512, 512, 248], dtype='uint8')
    for j in range(0 + i*248, 247 + i*248):
        single_image_name = image_arr[j]
        img_as_img = Image.open(single_image_name)
        img_as_np = np.asarray(img_as_img)
        #img_as_np1 = img_as_np.T
        allImg[:, :, j-i*248] = img_as_np
    allImg = np.transpose(allImg, [2, 0, 1])
    save_array_as_nii_volume(allImg, r'G:\csvd\segunet\epvs2930nii/pre_' + str(i + 1) + '.nii.gz')