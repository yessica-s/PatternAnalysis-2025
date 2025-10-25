import torch
from modules import uNet, DiceLoss
from dataset import load_images, get_data_loaders
from train import train
from utils import show_predictions, show_epoch_predictions 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing
train_tensor_set, validation_tensor_set, test_tensor_set, train_loader, validation_loader, test_loader = get_data_loaders(batch_size=4)
print("Data loaded successfully.")

# Improved uNet Model
model = uNet(in_channels=1, out_channels=4).to(device)

# Train the model
print("Training started...")
train(model, train_loader, validation_loader, validation_tensor_set, epochs=50, lr=1e-4)

# Run model on test set
model.eval()
dice_loss = DiceLoss()
test_loss = 0

with torch.no_grad():
    for image, mask in test_loader:
        image, mask = image.to(device), mask.to(device)
        output = model(image)
        loss = dice_loss(output, mask)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Dice Loss: {avg_test_loss:.4f}")

# Visualize predictions
print("Show predictions on test samples...")
show_predictions(model, test_tensor_set, device, num_classes=4, n=3)

print("Script complete.")
