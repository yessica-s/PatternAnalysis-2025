import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from modules import DiceLoss # import model from modules.py

print("Training model started.")

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

                                                           # lr=0.001
def train(model, train_loader, validation_loader, epochs=50, lr=1e-4, visualize_every=1):
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training Loop
        for batch_idx, (image, mask) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)

            optimizer.zero_grad()
            outputs








print("Training model complete.")