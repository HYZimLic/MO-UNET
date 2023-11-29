# MO-UNET
A fully automated, multi-output deep learning algorithm for multiple lesion segmentation.

In the diagnosis of chronic CSVD, cranial MRI imaging methods are preferred, including T2-FLAIR, SWI, and T2 sequences, which correspond to white matter hyperintensity, cerebral microbleeds, lacunae, and perivascular spaces, respectively. In this study, we aimed to develop a fully automatic, multi-output multi-lesion recognition and segmentation algorithm that can be used to detect and segment four types of cerebral small vessel diseases on multi-sequence MRI brain images.

Directory
- base/: Network basic module
- checkpoints/: Trained model weights
- encoders/: The encoder structure of segmounet architecture
- preprocessing/: Preprocessing and postprocessing files
- utils/: Auxiliary tool functions or classes
- unet/: The decoder structure and overall architecture of segmounet architecture
- evaluation/: Segmentation test related function
- predict.py: Test file
- train.py: Train file

Environment settings
- segmentation-models-pytorch==0.3.1
- torch==1.8.1+cu111

Data set
-Training: The training folder contains .npy files with corresponding names under the img and mask files.

  -train
    --img
      ---1.npy
      ---2.npy
      ---3.npy ...
      
    --mask
      ---1.npy
      ---2.npy
      ---3.npy ...
- Testing
  
  -test
    --img
      ---1.npy
      ---2.npy
      ---3.npy ...




