import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def vis_loss_psnr(metric, metrics):
    plt.figure(figsize=(10, 7))
    plt.plot(metrics[metric]["train"], color='red', label=f'train {metric.upper()}')
    plt.plot(metrics[metric]["val"], color='blue', label=f'val {metric.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.title(f'{metric.upper()} for all epochs')
    plt.legend()
    plt.show()
    
def vis_img(data_dir, lr, hr, pred, sample_psnr):
    print("PSNR:", sample_psnr)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.title("LR");   plt.imshow(to_pil_image(lr));  plt.axis('off')
    plt.subplot(1,3,2); plt.title("PRED"); plt.imshow(to_pil_image(pred));plt.axis('off')
    plt.subplot(1,3,3); plt.title("HR");   plt.imshow(to_pil_image(hr));  plt.axis('off')
    plt.tight_layout()
    plt.show()