from evaluation.metrics import *
import numpy as np
import nibabel as nib

filename = 'dice.txt'
data = 'cmb' #'whm','lacune','evps'
#If filename does not exist, it will be created automatically, 'w' means write data, before writing, it will clear the original data in the file!
with open(filename, 'a+') as f:  
    f.write('\npre_{}\t'.format(data))


for i in range(25):
    # Load nii.gz for predictive labeling here
    pred_seg_vol = nib.load(r'G:\csvd\cmb\cmbnii/pre_' + str(i + 1) + '.nii.gz')  
    pred = pred_seg_vol.get_fdata()
    # Load the gold standard here nii.gz
    gt_seg_vol = nib.load(r'G:\csvd\cmb\cmbgt/mask_' + str(i + 1) + '.nii.gz')  # 此处载入金标准nii.gz
    gt = gt_seg_vol.get_fdata()

    confusion_matrix = ConfusionMatrix(pred, gt)
    DSC = dice(pred, gt, confusion_matrix)
    Precision = precision(pred, gt, confusion_matrix)
    SE = sensitivity(pred, gt, confusion_matrix)
    SP = specificity(pred, gt, confusion_matrix)
    # JI = jaccard(pred, gt, confusion_matrix)
    # FPR = false_positive_rate(pred, gt, confusion_matrix)
    # FNR = false_negative_rate(pred, gt, confusion_matrix)
    # FRR = false_discovery_rate(pred, gt, confusion_matrix)
    # ASD = avg_surface_distance(pred, gt, confusion_matrix, voxelspacing=[0.488281, 0.488281, 7.2])
    # HD95 = hausdorff_distance_95(pred, gt, confusion_matrix, voxelspacing=[0.488281, 0.488281, 7.2])
    
    # If filename does not exist, it will be created automatically, 'w' means write data, before writing, it will clear the original data in the file!
    with open(filename, 'a+') as f:  
        f.write('\n{:.6f} \t{:.6f} \t{:.6f} \t{:.6f}'.format(
                    DSC, Precision, SE, SP))
        print('the metrics of {} is {:.6f}'.format(i, DSC))
