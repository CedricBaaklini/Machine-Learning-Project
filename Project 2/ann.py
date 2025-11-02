import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# --- Loss Function --- Marco
def make_loss(name = "ce", class_weights = None):
    """
    Creates and returns a specified loss function.
    Default: CrossEntropyLoss for multi-class classification.
    """
    name = name.lower()

    if name == "ce":
        return nn.CrossEntropyLoss(weight = class_weights)
    raise ValueError(f"Unknown loss '{name}'")

# --- Optimization Function --- Janelle
def optimize_mnist(model, dataloader, optimizer, loss_fn_placeholder, device='cpu'):
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

    avg_loss = total_loss / len(dataloader)
    
    return avg_loss

torch.manual_seed(42)

# --- Hyperparameters ---
batch_size = 256
lr = 1e-4
num_epochs = 70

# --- Data Preprocessing ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

# --- Neural Network Architecture ---
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 10))
        # includes dropout layers for regularization
    
    def forward(self, x):
        return self.net(x)

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, device='cpu'):
    """
    Evaluates model performance on a given dataset loader.
    Returns average loss and classification accuracy.
    """
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_total += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    avg_loss = loss_total / total
    accuracy = correct / total
    return avg_loss, accuracy

# --- Main Function ---
def main():
    """
    Main training function: loads data, trains the model, evaluates, and saves results.
    """
    loss_name = "ce"  # Cross Entropy Loss
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

    #model.to(device)
    model = SimpleANN().to(device)
    criterion = make_loss("ce")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training Loop with Early Stopping - Noah
    # --- Early Stopping Setup ---
    best_val_loss = float('inf')
    patience = 5 # Number of epochs to wait for validation loss improvement
    counter = 0

    # Trainign Loop - Noah
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = optimize_mnist(model, train_loader, optimizer, criterion, device=device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device=device)

        epoch_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}%")

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")  # Save best model
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load the best model before testing
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device=device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc * 100:.2f}%")

    # ---- Save Plots ----
    out_dir = Path("reports")
    out_dir.mkdir(parents = True, exist_ok= True)

    plt.figure()
    plt.plot(epoch_losses, label = "train")
    plt.plot(val_losses, label = "val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss ({loss_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"loss_curve_{loss_name}.png", dpi = 150)

    plt.figure()
    plt.plot([a * 100 for a in val_accs])
    plt.xlabel("Epoch")
    plt.ylabel("Val Acc (%)")
    plt.title(f"Val Accuracy ({loss_name})")
    plt.tight_layout()
    plt.savefig(out_dir / f"val_acc_{loss_name}.png", dpi = 150)

if __name__ == "__main__":
    main()