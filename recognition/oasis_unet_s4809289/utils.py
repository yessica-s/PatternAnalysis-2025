
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F

def denormalize_image(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # If grayscale, just clamp between 0–1
    if tensor.ndim == 3 and tensor.shape[0] == 1: # if grayscale, convert to 3D tensor
        return tensor.squeeze(0)
    else:
        return tensor

# Predictions during training
def show_epoch_predictions(model, dataset, epoch, n=3):
    # Show model predictions for binary segmentation after a specific epoch.

    print(f"Prediction After Epoch {epoch}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    show_predictions(model, dataset, device=next(model.parameters()).device, n=n)

def show_predictions(model, dataset, device, n=5):
    # Show model predictions vs ground truth on test dataset

    model.eval()
    plt.figure(figsize=(12, n * 3))

    indices = torch.randperm(len(dataset))[:n]
    with torch.no_grad():
        for i, idx in enumerate(indices, 1):
            image, mask = dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred_mask = (output >= 0.5).float()

            image_vis = denormalize_image(image[0]).cpu().numpy()
            mask_vis = mask.squeeze().cpu().numpy()
            pred_vis = pred_mask.squeeze().cpu().numpy()

            # Plot input, ground truth, prediction
            plt.subplot(n, 3, (i - 1) * 3 + 1)
            plt.imshow(image_vis, cmap='gray')
            plt.title("Input")
            plt.axis('off')

            plt.subplot(n, 3, (i - 1) * 3 + 2)
            plt.imshow(mask_vis, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')

            plt.subplot(n, 3, (i - 1) * 3 + 3)
            plt.imshow(pred_vis, cmap='gray')
            plt.title("Prediction")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

