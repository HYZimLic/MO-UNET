import os
import SimpleITK
import pydicom
import numpy as np
import cv2
from tqdm import tqdm

def is_dicom_file(filename):

    #判断某文件是否是dicom格式的文件
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def load_patient(src_dir):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = pydicom.read_file(src_dir + '/' + s)
            slices.append(instance)

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # new_spacing = [1.0, 1.0, 1.0]
    # size = np.array(image.GetSize())
    # spacing = np.array(image.GetSpacing())
    # new_spacing = np.array(new_spacing)
    # new_size = size * spacing / new_spacing
    # new_spacing_refine = size * spacing / new_size
    # new_spacing_refine = [float(s) for s in new_spacing_refine]
    # new_size = [int(s) for s in new_size]
    #
    # resample = SimpleITK.ResampleImageFilter()
    # resample.SetOutputDirection(image.GetDirection())
    # resample.SetOutputOrigin(image.GetOrigin())
    # resample.SetSize(new_size)
    # resample.SetOutputSpacing(new_spacing_refine)
    # resample.SetInterpolator(SimpleITK.sitkNearestNeighbor)
    # newimage = resample.Execute(image)

    newimage = SimpleITK.GetArrayFromImage(image)
    newimage[newimage < 0] = 0
    #img_array = (newimage - np.min(newimage))/(np.max(newimage) - np.min(newimage))
    return newimage


if __name__ == '__main__':
	#dicom文件目录
    dicom_dir = r'F:\EPVS\org/'
    img_names = os.listdir(dicom_dir)
    for img_name in img_names:
    # 读取dicom文件的元数据(dicom tags)
        slices = load_patient(dicom_dir + img_name)
        print('The number of dicom files : ', len(slices))
        # 提取dicom文件中的像素值
        image = get_pixels_hu_by_simpleitk(dicom_dir + img_name)
        for i in tqdm(range(image.shape[0])):
            #输出png文件目录
            img_path = r"F:\EPVS\img/"
            org_img = image[i]
            #org_img[0, 0] = 2
            float_arr = org_img.astype(np.float64)
            #
            #org_img =image[i]*255

            # height, width = image.shape[1], image.shape[2]
            # cropped_img = org_img[height // 2 - 500 // 2:height // 2 + 500 // 2, width // 2 - 500 // 2:width // 2 + 500 // 2]
            # 保存图像数组
            np.save(img_path + 'EPVS' + '_' + str(img_name).split('.')[0] + '_' + str(i + 1).rjust(3, '0') + '.npy', float_arr)
            #np.save(img_path + str(img_name).split('.')[0] + '.npy', float_arr)
            #cv2.imwrite(img_path, org_img)
            #cv2.imwrite(img_path, cropped_img)