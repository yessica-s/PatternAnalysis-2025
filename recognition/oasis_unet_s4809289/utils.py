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

def show_epoch_predictions(model, dataset, epoch, n=3):
    # Show model predictions for multi-class segmentation after a specific epoch.
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(f'Prediction After Epoch {epoch}', fontsize=16, fontweight='bold')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define colors for up to 4 classes
    cmap = ListedColormap(['black', 'red', 'green', 'blue'])

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]
    
            image_input = image.unsqueeze(0).to(device)

            # Get predicted mask
            pred = model(image_input)

            if pred.shape[1] > 1:  
                pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()
            else:
                pred_mask = (torch.sigmoid(pred[0, 0]) > 0.5).cpu().numpy().astype(int)

            true_mask_np = true_mask.numpy()

            # Denormalize image for display
            img_show = denormalize_image(image)

            if img_show.ndim == 3 and img_show.shape[0] == 1:
                img_display = img_show.squeeze(0).numpy()
            elif img_show.ndim == 3 and img_show.shape[0] > 1:
                img_display = img_show.permute(1, 2, 0).numpy()
            elif img_show.ndim == 2:
                img_display = img_show.numpy()

            # Plot Image
            axes[0, i].imshow(img_display, cmap='gray' if img_display.ndim == 2 else None)
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            # Plot ground truth mask
            # Check masks have shape (H, W) for matplotlib
            if true_mask_np.ndim == 3 and true_mask_np.shape[0] == 1:
                true_mask_np = true_mask_np.squeeze(0)

            im1 = axes[1, i].imshow(true_mask_np, cmap=cmap, vmin=0, vmax=3)
            axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
            axes[1, i].axis('off')

            # Plot predicted
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask.squeeze(0)

            im2 = axes[2, i].imshow(pred_mask, cmap=cmap, vmin=0, vmax=3)
            acc = np.mean(pred_mask == true_mask_np)
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {acc:.3f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    model.train()

                                            # num_classes = 4
def show_predictions(model, dataset, device, num_classes=1, n=3, title="Multiclass Segmentation Results"):
    # Show model predictions vs ground truth on test dataset
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    with torch.no_grad(): # no gradient descent/backpropagation
        for i in range(n):
            image, true_mask = dataset[i]
            image = image.unsqueeze(0).to(device) 

            # Get probabilities
            output = model(image)

            # sigmoid for binary, softmax for multi-class
            if num_classes > 1:
                pred_mask = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu()
            else:
                pred_mask = (torch.sigmoid(output) > 0.5).int().squeeze().cpu()

            img_show = image.squeeze(0).cpu()
            if img_show.ndim == 3 and img_show.shape[0] == 1:
                img_np = img_show.squeeze(0).numpy()
            elif img_show.ndim == 3 and img_show.shape[0] > 1:
                img_np = img_show.permute(1, 2, 0).numpy()
            elif img_show.ndim == 2:
                img_np = img_show.numpy()

            true_mask = true_mask.squeeze().cpu().numpy()
            pred_mask = pred_mask.numpy()

            # Original image
            axes[0, i].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            axes[0, i].set_title(f"Original {i+1}", fontweight='bold')
            axes[0, i].axis('off')

            # Ground truth mask
            axes[1, i].imshow(true_mask, cmap='tab10', vmin=0, vmax=num_classes-1)
            axes[1, i].set_title("Ground Truth", fontweight='bold')
            axes[1, i].axis('off')

            # Predicted mask
            axes[2, i].imshow(pred_mask, cmap='tab10', vmin=0, vmax=num_classes-1)
            acc = np.mean(pred_mask == true_mask)
            axes[2, i].set_title(f"Prediction (Acc: {acc:.2f})", fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
