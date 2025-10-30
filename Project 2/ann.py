"""
Project 2 - Artificial Neural Network using MNIST dataset
Description:
   This script trains a simple feedforward neural network on the MNIST dataset.
   It implements Cross-Entropy Loss for multi-class classification using PyTorch.
   The model is trained for 20 epochs and outputs two plots:
       - loss_curve_CE.png (training and validation loss)
       - val_acc_CE.png (validation accuracy)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib2 import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# --- Loss Function ---
def make_loss(name="ce", class_weights=None):
    """
    Creates and returns a specified loss function.
    Default: CrossEntropyLoss for multi-class classification.
    """
    name = name.lower()

    if name == "ce":
        return nn.CrossEntropyLoss(weight = class_weights)
    raise ValueError(f"Unknown loss '{name}'")

torch.manual_seed(42)

# Hyperparameters
# Batch Sampling Size - How many samples per batch
batch_size = 128
# Learning Rate - Weights updated during training
lr = 1e-3
# Epoch # - Times entire training dataset is passed through the mode
num_epochs = 20

# --- Data Preprocessing ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

# --- Neural Network Architecture ---
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(inplace=True), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 10))
        
    def forward(self, x):
        return self.net(x)

# ---- Setup ----
model = SimpleANN()

loss_name = "CE" # Cross-Entropy Loss
criterion = make_loss(loss_name)
optimizer = optim.Adam(model.parameters(), lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluation Function ---
def evaluate(loader):
    """
    Evaluates model performance on a given dataset loader.
    Returns average loss and classification accuracy.
    """
    model.eval()
    correct, total, loss_sum, batches = 0, 0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item()
            batches += 1
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return loss_sum / max(1, batches), correct / max(1, total)

def optimize_epoch(model, dataloader, optimizer, loss_fn_placeholder, device='cpu'):
    """
    Perform one training epoch on MNIST using a placeholder loss function.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    #images and labels in dataloader:
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        #Forward pass
        outputs = model(images)

        #Compute loss using placeholder
        loss = loss_fn_placeholder(outputs, labels)

        #Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    """
    Main training function: loads data, trains the model, evaluates, and saves results.
    """
    num_workers = 0
    pin_memory = torch.cuda.is_available()
    epoch_losses, val_losses, val_accs = [], [], []

    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_size = 50_000
    val_size = 10_000
    train_set, val_set = random_split(train_full, [train_size, val_size], torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size, False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size, False, num_workers=num_workers, pin_memory=pin_memory)

    model.to(device)

    # Lists for plotting
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # Training Loop - Noah
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, steps = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            
        train_loss = running_loss / max(1, steps)
        val_loss, val_acc = evaluate(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "f"train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}%")

    test_loss, test_acc = evaluate(test_loader)
    print(f"Test: loss={test_loss:.4f} acc={test_acc*100:.2f}%")

    # Plotting - Noah
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_results.png") 

if __name__ == "__main__":
    main()