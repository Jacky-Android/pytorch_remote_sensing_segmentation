import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from transform import *
import matplotlib.patches as mpatches
from PIL import Image
import random
from skimage import io
from torchvision import transforms 
#from torch import functional as F
from torchvision.transforms import functional as F



CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)

'''ORIGIN_IMG_SIZE = (1000, 1000)
INPUT_IMG_SIZE = (1000, 1000)
TEST_IMG_SIZE = (1000, 1000)'''

def get_training_transform():
    train_transform = [
       
        albu.Resize(1024,1024),
        albu.RandomCrop(512, 512),
        # 水平翻转
        albu.HorizontalFlip(p=0.5),
        # 随机旋转
        albu.Rotate(limit=30, p=0.5),
        # 随机缩放
        # albu.RandomScale(scale_limit=0.2, p=0.5),
        # 随机亮度和对比度调整
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        # 随机Gamma调整
        
        albu.Normalize()

    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    #img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Resize(1024,1024),
        albu.Normalize(),
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask



class PotsdamDataset(Dataset):
    def __init__(self, data_root='G:/postam_orignal/kaggle/working', mode='val', img_dir='images/test', mask_dir='anns/test',
                 img_suffix='.tif', mask_suffix='.tif', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if self.mode == 'val' :
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.load_img_and_mask(index)
                img, mask = self.transform(img, mask)
        elif self.mode == 'train' :
            img, mask = self.load_img_and_mask(index)
            #img, mask = self.load_mosaic_img_and_mask(index)
            img, mask = train_aug(img, mask)
            '''if self.transform:
                img, mask = self.transform(img, mask)'''

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        return img, mask