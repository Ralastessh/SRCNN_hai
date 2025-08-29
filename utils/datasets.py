import os
import random
import shutil

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils.getLH import low_resolution, high_resolution

class DIV2K(Dataset):
    def __init__(self, img_dir):
        super().__init__()
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))

    def __getitem__(self, idx):
        filename = self.img_list[idx]
        
        # YCbCr의 Y 채널 사용
        ycbcr_lr = low_resolution(self.img_dir, filename).convert('YCbCr')
        ycbcr_hr = high_resolution(self.img_dir, filename).convert('YCbCr')
        y_lr, cb_lr, cr_lr = ycbcr_lr.split()
        y_hr, cb_hr, cr_hr = ycbcr_hr.split()

        totensor = ToTensor()
        y_lr = totensor(y_lr)
        y_hr = totensor(y_hr)
        
        return y_lr, y_hr

    def __len__(self):
        return len(self.img_list)
    
def split_ttv(img_dir, num_train, num_test):
    img_list = sorted([
        f for f in os.listdir(img_dir)
        if os.path.isfile(os.path.join(img_dir, f))
    ])
    total = len(img_list)

    random.shuffle(img_list)
    
    train_img = img_list[:num_train]
    test_img = img_list[num_train:num_train + num_test]
    val_img = img_list[num_train + num_test:]
    
    for split, files in zip(['train', 'test', 'val'], [train_img, test_img, val_img]):
        dest_dir = os.path.join(img_dir, split)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(img_dir, f), os.path.join(dest_dir, f))