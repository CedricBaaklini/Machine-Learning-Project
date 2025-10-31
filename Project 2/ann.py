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
from matplotlib import pyplot as plt
from pathlib2 import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def optimize_epoch(model, dataloader, optimizer, loss_fn_placeholder, device='cpu'):
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

torch.manual_seed(42)

# Hyperparameters
# Batch Sampling Size - How many samples per batch
batch_size = 128
# Learning Rate - Weights updated during training
lr = 1e-3
# Epoch # - Times the entire training dataset is passed through the mode
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

optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
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

def main():
    """
    Main training function: loads data, trains the model, evaluates, and saves results.
    """
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load MNIST
    train_full = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)

    # Split train into train/val
    val_size = 10_000
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(1, num_epochs + 1):
        tr_loss = optimize_epoch(model, train_loader, optimizer, criterion, device=device)
        val_loss, val_acc = evaluate(val_loader)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch:02d}/{num_epochs}  |  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # Final test eval
    test_loss, test_acc = evaluate(test_loader)
    print(f"Test  |  loss={test_loss:.4f}  acc={test_acc:.4f}")

    # ----- Graphs -----
    # 1) Training/Validation Loss
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend(); plt.tight_layout(); plt.savefig("loss_curve.png", dpi=150); plt.close()

    # 2) Validation Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.ylim(0,1)
    plt.tight_layout(); plt.savefig("val_accuracy.png", dpi=150); plt.close()

    # 3) Confusion Matrix (on test set)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.numpy().tolist())
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (Test acc={test_acc:.3f})")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

    # 4) Prediction Grid (first 36 test images)
    images, labels = next(iter(test_loader))
    images, labels = images[:36], labels[:36]
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device))
        preds = logits.argmax(dim=1).cpu()
    images = images.squeeze(1).numpy()  # (N, 28, 28)

    fig, axes = plt.subplots(6, 6, figsize=(8,8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"P:{preds[i].item()} T:{labels[i].item()}", fontsize=8)
        ax.axis("off")
    plt.suptitle("Sample Predictions (Test)", y=0.92)
    plt.tight_layout(); plt.savefig("sample_predictions.png", dpi=150); plt.close()

    # Save model
    torch.save(model.state_dict(), "mnist_simple_ann.pt")

if __name__ == "__main__":
    main()