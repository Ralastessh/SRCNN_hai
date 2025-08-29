# SRCNN
## Project Structure
```
SRCNN/
├── SR_outputs/              # Super-resolution result images (e.g., *_pred.png)
│   └── [*.png]
├── checkpoints/             # Model checkpoint files (.pth)
│   └── srcnn_checkpoint.pth
├── data/                    # Input datasets
├── models/
│   └── srcnn.py             # SRCNN model architecture
├── utils/
│   ├── datasets.py          # Custom dataset class
│   ├── getLH.py             # Y channel conversion or cropping
│   ├── metrics.py           # PSNR
│   └── visualization.py     # Functions for plotting or image comparison
├── main.py                  # Training & inference entry point
├── train.py                 # Training script
└── test.py                  # Evaluation / prediction script
```

## How to Run
```bash
python main.py
```
* You have to add the datasets to 'data/DIV2K_519sampled'
## Result
### Loss
<img width="1000" height="700" alt="Lossforallepochs" src="https://github.com/user-attachments/assets/d2b26f8d-a124-4619-acd3-62aa0ae7f08a" />

### PSNR
<img width="1000" height="700" alt="psnrforallepochs" src="https://github.com/user-attachments/assets/66922c90-6a79-46e3-9e06-f3e92174d74e" />

### Images
<img width="1200" height="430" alt="Figure_1" src="https://github.com/user-attachments/assets/761d6958-ac0f-44cd-b5ee-f66e0c541a3a" />
<img width="1200" height="432" alt="Figure_2" src="https://github.com/user-attachments/assets/3fde3fb4-5b54-4e39-add3-8fec77241e03" />
