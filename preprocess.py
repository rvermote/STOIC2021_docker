import torch
import numpy as np
import SimpleITK as sitk
import math as m
import cv2
from typing import Iterable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import device

def preprocess(input_image):
    
    #input_image = sitk.ReadImage("/input/6.mha")
    input_image = sitk.GetArrayFromImage(input_image)
    
    #get cropped slice from CT scan
    img_slice1 = input_image[m.ceil(0.6*input_image.shape[0]), 115:435, 45:511-46]
    img_slice2 = input_image[m.ceil(0.7*input_image.shape[0]), 115:435, 45:511-46]
    img_slice3 = input_image[m.ceil(0.5*input_image.shape[0]), 115:435, 45:511-46]
    
    #normalize to [0,1]
    clip_min = -1024
    clip_max = 0
    img_slice1 = np.clip(img_slice1, clip_min, clip_max)
    img_slice1 = (img_slice1 - clip_min) / (clip_max - clip_min)
    img_slice2 = np.clip(img_slice2, clip_min, clip_max)
    img_slice2 = (img_slice2 - clip_min) / (clip_max - clip_min)
    img_slice3 = np.clip(img_slice3, clip_min, clip_max)
    img_slice3 = (img_slice3 - clip_min) / (clip_max - clip_min)

    #restore aspect ratio
    s = max(img_slice1.shape[0:2])
    f1 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice1.shape[1])//2,(s - img_slice1.shape[0])//2
    f1[ay:img_slice1.shape[0]+ay,ax:ax+img_slice1.shape[1]] = img_slice1
    
    s = max(img_slice2.shape[0:2])
    f2 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice2.shape[1])//2,(s - img_slice2.shape[0])//2
    f2[ay:img_slice2.shape[0]+ay,ax:ax+img_slice2.shape[1]] = img_slice2
    
    s = max(img_slice3.shape[0:2])
    f3 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice3.shape[1])//2,(s - img_slice3.shape[0])//2
    f3[ay:img_slice3.shape[0]+ay,ax:ax+img_slice3.shape[1]] = img_slice3
    
    input_image1 = cv2.merge([f1,f1,f1])
    input_image2 = cv2.merge([f2,f2,f2])
    input_image3 = cv2.merge([f3,f3,f3])
    
    val_transforms = A.Compose(
        [
            A.Resize(width=224, height=224),
            ToTensorV2(),
        ],
        additional_targets={'image2': 'image', 'image3': 'image'}
    )
    
    trans = val_transforms(image = input_image1, image2=input_image2, image3=input_image3)
    input_image1 = trans["image"]
    input_image2 = trans["image2"]
    input_image3 = trans["image3"]
    input_image1 = input_image1.unsqueeze(0)
    input_image2 = input_image2.unsqueeze(0)
    input_image3 = input_image3.unsqueeze(0)
    input_image1 = input_image1.to(device, dtype=torch.float)
    input_image2 = input_image2.to(device, dtype=torch.float)
    input_image3 = input_image3.to(device, dtype=torch.float)


    return input_image1, input_image2, input_image3


if __name__ == "__main__":
    import sys

    print('loading')
    input_image = sitk.ReadImage(sys.argv[1])
    input_image = sitk.GetArrayFromImage(input_image)
    
    #get cropped slice from CT scan
    img_slice1 = input_image[m.ceil(0.6*input_image.shape[0]), 115:435, 45:511-46]
    img_slice2 = input_image[m.ceil(0.7*input_image.shape[0]), 115:435, 45:511-46]
    img_slice3 = input_image[m.ceil(0.5*input_image.shape[0]), 115:435, 45:511-46]
    
    #normalize to [0,1]
    clip_min = -1024
    clip_max = 0
    img_slice1 = np.clip(img_slice1, clip_min, clip_max)
    img_slice1 = (img_slice1 - clip_min) / (clip_max - clip_min)
    img_slice2 = np.clip(img_slice2, clip_min, clip_max)
    img_slice2 = (img_slice2 - clip_min) / (clip_max - clip_min)
    img_slice3 = np.clip(img_slice3, clip_min, clip_max)
    img_slice3 = (img_slice3 - clip_min) / (clip_max - clip_min)

    #restore aspect ratio
    s = max(img_slice1.shape[0:2])
    f1 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice1.shape[1])//2,(s - img_slice1.shape[0])//2
    f1[ay:img_slice1.shape[0]+ay,ax:ax+img_slice1.shape[1]] = img_slice1
    
    s = max(img_slice2.shape[0:2])
    f2 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice2.shape[1])//2,(s - img_slice2.shape[0])//2
    f2[ay:img_slice2.shape[0]+ay,ax:ax+img_slice2.shape[1]] = img_slice2
    
    s = max(img_slice3.shape[0:2])
    f3 = np.zeros((s,s),np.float32)
    ax,ay = (s - img_slice3.shape[1])//2,(s - img_slice3.shape[0])//2
    f3[ay:img_slice3.shape[0]+ay,ax:ax+img_slice3.shape[1]] = img_slice3
    
    input_image1 = cv2.merge([f1,f1,f1])
    input_image2 = cv2.merge([f2,f2,f2])
    input_image3 = cv2.merge([f3,f3,f3])
    
    val_transforms = A.Compose(
        [
            A.Resize(width=224, height=224),
            ToTensorV2(),
        ],
        additional_targets={'image2': 'image', 'image3': 'image'}
    )
    
    trans = val_transforms(image = input_image1, image2=input_image2, image3=input_image3)
    input_image1 = trans["image"]
    input_image2 = trans["image2"]
    input_image3 = trans["image3"]
    input_image1 = input_image1.unsqueeze(0)
    input_image2 = input_image2.unsqueeze(0)
    input_image3 = input_image3.unsqueeze(0)
    input_image1 = input_image1.to(device, dtype=torch.float)
    input_image2 = input_image2.to(device, dtype=torch.float)
    input_image3 = input_image3.to(device, dtype=torch.float)


    print('done.')
