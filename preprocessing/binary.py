import glob
import numpy as np
from PIL import Image
import cv2
import os
image_path = r'G:\csvd\segunet\epvs7241/'

saveimg_path = r'G:\csvd\segunet\epvs7241er/'
img_names = os.listdir(image_path)

for i in range(len(img_names)):
    img_name = img_names[i]
    img_as_img = Image.open(image_path + img_name)
    img_as_np = np.asarray(img_as_img)
    retval, dst = cv2.threshold(img_as_np, 80, 255, cv2.THRESH_BINARY)
    #dst1 = dst/255
    #print(dst)
    # cv2.imwrite(saveimg_path + str(img_name).split('.')[0] + '.png', dst)
    cv2.imwrite(saveimg_path + img_name, dst)