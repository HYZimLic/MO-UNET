import nibabel as nib
import os
import numpy as np

# img_path = r'F:\11bishe\nii\houcengneimo\neimo_Imagetest_nii_gz512/'
seg_path = r'G:\csvd\xinEPVS segmentation/'
# saveimg_path = r'F:\11bishe\neimohouceng\dan\testimgnpy/'
saveseg_path = r'G:\csvd\epvs1\mask/'

# img_names = os.listdir(img_path)
seg_names = os.listdir(seg_path)


# for img_name in img_names:
#     print(img_name)
#     img = nib.load(img_path + img_name).get_data() #载入
#     img = np.array(img)
#     ceng1 = np.shape(img)[0]
#     for i in range(ceng1):
#         img1 = img[i, :, :]
#         np.save(saveimg_path + str(img_name).split('.')[0] + '_' + str(i+1).rjust(2,'0') + '.npy', img1) #保存


for seg_name in seg_names:
    print(seg_name)
    seg = nib.load(seg_path + seg_name).get_data()
    seg = np.array(seg)
    ceng2 = np.shape(seg)[2]
    for j in range(ceng2):
        img2 = seg[:, :, j]
        # img2[img2 == 4] = 0
        img2 = img2.T
        float_arr = img2.astype(np.float64)
        np.save(saveseg_path + 'EPVS' + '_' + str(seg_name).split('.')[0] + '_' + str(j + 1).rjust(3, '0') + '.npy', float_arr)
        #np.save(saveseg_path + '50' +'_'+ str(seg_name).split('.')[0] + '.npy', img2)