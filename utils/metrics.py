import torch

def psnr(gt, output):
    return 10 * torch.log10(1.0 / torch.mean((gt - output) ** 2))