import torch
from torchvision.transforms import CenterCrop, Resize, Compose
from PIL import Image
import os

def preprocessing(img_dir, img_name, transform):
    try:
        src = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
        processed = transform(src)
        return processed    
    except Exception as e:
        print('Error:', e)

def low_resolution(img_dir, img_name):
    transform = Compose([
                CenterCrop(64),
                Resize(32, interpolation=Image.BICUBIC),
                Resize(64, interpolation=Image.BICUBIC)
            ])
    return preprocessing(img_dir, img_name, transform)

def high_resolution(img_dir, img_name):
    transform = CenterCrop(64)
    return preprocessing(img_dir, img_name, transform)
