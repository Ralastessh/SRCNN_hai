import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.srcnn import SRCNN
from utils.datasets import DIV2K, split_ttv
from utils.metrics import psnr

def train_srcnn(data_dir, device):
    # 모델 호출, 학습률, Epoch, batch size, optimizer, loss 정의
    model = SRCNN().to(device)
    
    lr = 1e-3
    epochs = 50
    batch_size = 32
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_criteria = nn.MSELoss()
    
    # 학습
    
    # 시각화를 위해 저장
    all_train_loss = []
    all_val_loss = []
    all_train_psnr = []
    all_val_psnr = []
    
    split_ttv(data_dir, 363, 104)

    train_set = DataLoader(DIV2K(data_dir + '/train'), batch_size=batch_size, shuffle=True, num_workers=4)
    val_set = DataLoader(DIV2K(data_dir + '/val'), batch_size=batch_size, shuffle=True, num_workers=4)

    for epoch in tqdm(range(1, epochs + 1), desc="Training Progress", total=epochs):
        # 모델을 학습 모드로 전환
        model.train()
        train_losses = []
        train_psnrs = []
        for y_lr, y_hr in train_set:
            train_pred = model(y_lr.to(device))
            train_loss = loss_criteria(train_pred, y_hr.to(device))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
            train_psnrs.append(psnr(y_hr.to(device), train_pred).item())
        all_train_loss.append(sum(train_losses) / len(train_losses))
        all_train_psnr.append(sum(train_psnrs) / len(train_psnrs))

        model.eval()
        val_losses = []
        val_psnrs = []
        with torch.no_grad():
            for y_lr, y_hr in val_set:
                val_pred = model(y_lr.to(device))
                val_loss = loss_criteria(val_pred, y_hr.to(device))
                val_losses.append(val_loss.item())
                val_psnrs.append(psnr(y_hr.to(device), val_pred).item())
        all_val_loss.append(sum(val_losses) / len(val_losses))
        all_val_psnr.append(sum(val_psnrs) / len(val_psnrs))

        if epoch % 10 == 0:
            tqdm.write(f'===> Epoch: {epoch}/{epochs}')
            tqdm.write(f'[train] loss: {sum(all_train_loss) / len(all_train_loss):.4f}, PSNR: {sum(all_train_psnr) / len(all_train_psnr):.2f}')
            tqdm.write(f'[val]   loss: {sum(all_val_loss) / len(all_val_loss):.4f}, PSNR: {sum(all_val_psnr) / len(all_val_psnr):.2f}')

    # 모델 가중치 저장
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/srcnn_checkpoint.pth')

    metrics = {
        "psnr": {"train": all_train_psnr, "val": all_val_psnr},
        "loss": {"train": all_train_loss, "val": all_val_loss}
    }
    return metrics