import torch
from modules import uNet, DiceLoss
from dataset import load_image, get_data_loaders
from train import train
from utils import show_predictions, show_epoch_predictions 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ## Data preprocessing
    train_tensor_set, validation_tensor_set, test_tensor_set, train_loader, val_loader, test_loader = get_data_loaders(batch_size=4)
    print("Data loaded successfully.")

    # Improved uNet Model
    model = uNet(in_channels=1, out_channels=1).to(device)  # out_channels = 1

    # Train the model
    print("Training started...")
    train(model, train_loader, val_loader, val_loader.dataset, epochs=50, lr=1e-4, visualize_every=5)
    print("Training complete.")

    # Run model on test set
    print("Running model on test set...")
    model.eval()
    dice_loss = DiceLoss()
    test_loss = 0
    test_coefficients = 0

    with torch.no_grad():
        for image, mask in test_loader:
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            loss = dice_loss(output, mask) # returns dice loss
            test_loss += loss.item()

            dice_coeffcient = 1 - loss.item() # get dice coefficient
            test_coefficients += dice_coeffcient

    avg_test_loss = test_loss / len(test_loader)
    avg_test_coefficient = test_coefficients / len(test_loader)
    print(f"Average Test Dice Loss: {avg_test_loss:.4f}")
    print(f"Average Test Dice Coefficient: {avg_test_coefficient:.4f}")

    # Visualize predictions
    print("Visualizing predictions on test samples...")
    show_predictions(model, test_loader.dataset, device, n=3)

    print("Script complete.")