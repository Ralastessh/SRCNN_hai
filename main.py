import torch
from tqdm import tqdm

from models.srcnn import SRCNN
from utils.datasets import DIV2K, split_ttv
from train import train_srcnn
from test import test_srcnn
from utils.metrics import psnr
from utils.visulaization import vis_loss_psnr, vis_img

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    
if __name__ == '__main__':
    print(f'Device: {device}\n')

    data_dir = "./data/DIV2K_519sampled" 

    # 학습
    metrics = train_srcnn(data_dir, device)
    vis_loss_psnr("loss", metrics)
    vis_loss_psnr("psnr", metrics)
    torch.mps.empty_cache()
    
    # 추론
    lr, hr, pred, psnr_val = test_srcnn(data_dir, device)
    vis_img(data_dir, lr, hr, pred, psnr_val)
    torch.mps.empty_cache()