import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def denormalize_image(tensor):
    """Safely denormalize an image tensor (works on CPU/GPU and for grayscale or RGB)."""
    # Move to CPU if necessary
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # If grayscale, just clamp between 0–1
    if tensor.shape[0] == 1:
        return torch.clamp(tensor, 0, 1)
    
    # If 3-channel RGB
    elif tensor.shape[0] == 3:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm_tensor = tensor * std + mean
        return torch.clamp(denorm_tensor, 0, 1)
    
    # For unexpected channel numbers
    else:
        # fallback: just clamp
        return torch.clamp(tensor, 0, 1)

def show_epoch_predictions(model, dataset, epoch, n=3):
    """Show model predictions for multi-class segmentation after a specific epoch."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(f'🎯 Predictions After Epoch {epoch}', fontsize=16, fontweight='bold')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define colors for up to 4 classes
    cmap = ListedColormap(['black', 'red', 'green', 'blue'])

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]
            
            image_input = image.unsqueeze(0).to(device)

            pred = model(image_input)

            # If softmax: take argmax; if sigmoid: threshold
            if pred.shape[1] > 1:  
                pred_mask = torch.argmax(pred, dim=1)[0].cpu().numpy()
            else:
                pred_mask = (pred[0, 0] > 0.5).cpu().numpy().astype(int)

            true_mask_np = true_mask.numpy()

            # Denormalize image for display
            img_show = denormalize_image(image)
            img_display = img_show.permute(1, 2, 0).numpy()

            # --- Plot Original ---
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            # --- Ground Truth ---
            im1 = axes[1, i].imshow(true_mask_np, cmap=cmap, vmin=0, vmax=3)
            axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
            axes[1, i].axis('off')

            # --- Prediction ---
            im2 = axes[2, i].imshow(pred_mask, cmap=cmap, vmin=0, vmax=3)
            acc = np.mean(pred_mask == true_mask_np)
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {acc:.3f})', fontweight='bold')
            axes[2, i].axis('off')

            # Optional: add colorbar only once
            if i == 0:
                plt.colorbar(im2, ax=axes[:, i], shrink=0.6, ticks=range(4), label='Class index')

    plt.tight_layout()
    plt.show()
    model.train()
