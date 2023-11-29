import glob
import numpy as np
image_path = r'G:\npy/'
image_arr = glob.glob(str(image_path) + str("/*"))
image_arr.sort()
allImg = []
allImg = np.zeros([512, 512, len(image_arr)])
for i in range(len(image_arr)):
    single_image_name = image_arr[i]
    img_as_img = np.load(single_image_name)
    img_as_np = np.asarray(img_as_img)
    allImg[:, :, i] = img_as_np
print(np.max(allImg))