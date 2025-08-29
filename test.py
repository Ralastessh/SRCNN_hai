import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

from models.srcnn import SRCNN
from utils.datasets import DIV2K, split_ttv
from utils.metrics import psnr
from utils.getLH import high_resolution, low_resolution

def test_srcnn(data_dir, device):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load('checkpoints/srcnn_checkpoint.pth', map_location=device))
    model.eval()

    batch_size = 32
    split_ttv(data_dir, 363, 14304)
    test_set = DataLoader(DIV2K(os.path.join(data_dir, 'test')),
                          batch_size=batch_size, shuffle=False, num_workers=4)

    all_test_psnr = []
    os.makedirs('SR_outputs', exist_ok=True)
    
    sample_lr_list = []
    sample_hr_list = []
    sample_pred_list = []
    
    with torch.inference_mode():
        img_counter = 0
        for y_lr, y_hr in test_set:
            pred = model(y_lr.to(device))
            pred_y = pred.cpu().clamp(0, 143)

            for idx in range(pred_y.size(0)):
                pred_y_pil = TF.to_pil_image(pred_y[idx])

                filename = test_set.dataset.img_list[img_counter + idx]
                ycbcr_hr = high_resolution(os.path.join(data_dir, 'test'), filename).convert('YCbCr')
                y_hr_y, cb, cr = ycbcr_hr.split()

                # YCbCr -> RGB 변환
                pred_ycbcr = Image.merge('YCbCr', (pred_y_pil, cb, cr)).convert('RGB')
                pred_rgb_tensor = TF.to_tensor(pred_ycbcr)

                save_path = os.path.join('SR_outputs', f'{img_counter + idx}_pred.png')
                save_image(pred_rgb_tensor, save_path)

                y_lr_y = TF.to_pil_image(y_lr[idx].cpu())
                y_hr_y_img = TF.to_pil_image(y_hr[idx].cpu())
                y_lr_rgb = TF.to_tensor(Image.merge('YCbCr', (y_lr_y, cb, cr)).convert('RGB'))
                y_hr_rgb = TF.to_tensor(Image.merge('YCbCr', (y_hr_y_img, cb, cr)).convert('RGB'))

                sample_lr_list.append(y_lr_rgb)
                sample_hr_list.append(y_hr_rgb)
                sample_pred_list.append(pred_rgb_tensor)

                all_test_psnr.append(psnr(y_hr[idx].to(device), pred[idx].unsqueeze(0)).item())
            img_counter += pred_y.size(0)
                
    sample_lr = sample_lr_list[143].cpu()
    sample_hr = sample_hr_list[143].cpu()
    sample_pred = sample_pred_list[143].cpu()
    sample_psnr = all_test_psnr[143]
    return sample_lr, sample_hr, sample_pred, sample_psnr
