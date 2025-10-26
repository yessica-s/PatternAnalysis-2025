import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from modules import DiceLoss 
from utils import denormalize_image, show_epoch_predictions

                                                           # lr=0.001
def train(model, train_loader, validation_loader, validation_tensor_set, epochs=50, lr=1e-4, visualize_every=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    loss_function = DiceLoss()
    bce_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    validation_loss = []
    train_dice_coeff = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_dice_coeff = 0 

        # Training Loop
        for batch_idx, (image, mask) in enumerate(train_loader):
            image, mask = image.to(device), mask.to(device)

            optimizer.zero_grad()
            outputs = model(image) # returns softmax output format

            # Loss function (DiceLoss) on all classes
            # loss = 0.5 * bce_function(outputs, mask) + 0.5 * loss_function(outputs, mask)
            loss = loss_function(outputs, mask)
            loss.backward()
            optimizer.step() # gradient descent

            epoch_train_loss += loss.item()
            epoch_dice_coeff += (1 - loss.item()) 

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss.append(avg_train_loss)

        avg_train_coeff = epoch_dice_coeff / len(train_loader)
        train_dice_coeff.append(avg_train_coeff)

        # Validation
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad(): # do not update gradients/do backpropagation
            for image, mask in validation_loader:
                image, mask = image.to(device), mask.to(device)
                outputs = model(image)
                # val_loss = 0.5 * bce_function(outputs, mask) + 0.5 * loss_function(outputs, mask)
                val_loss = loss_function(outputs, mask)
                epoch_val_loss += val_loss.item()

            avg_val_loss = epoch_val_loss / len(validation_loader)
            validation_loss.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}  |  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice Coeff: {avg_train_coeff:.4f}")

        if (epoch + 1) % visualize_every == 0:
            show_epoch_predictions(model, validation_tensor_set, epoch+1)

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Training model complete.")
    return train_loss, validation_loss