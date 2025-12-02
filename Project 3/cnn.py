import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Hyperparameters (for tuning) ---
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.001
DROPOUT = 0.35
VAL_SPLIT = 0.1
RANDOM_SEED = 42
WEIGHT_DECAY = 1e-4

# --- Data Augmentation and Normalization ---
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter
    transforms.RandomRotation(15),  # Random rotation up to 15 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# --- Dataset Loading and Splitting ---
def load_datasets():
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    val_size = int(VAL_SPLIT * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    return train_set, val_set, test_set

# --- Visualization ---
def visualize_samples(dataset, classes):
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(loader))
    images = images[:8]
    labels = labels[:8]
    fig, axes = plt.subplots(1, 8, figsize=(16, 2))
    for i in range(8):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip((img * np.array([0.2023, 0.1994, 0.2010])) + np.array([0.4914, 0.4822, 0.4465]), 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
    plt.tight_layout()
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/sample_images.png")
    plt.close()

# --- CNN Model ---
class ConvNetCIFAR10(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# --- Training and Evaluation ---
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return running_loss / total, correct / total

# --- Main ---
def main():
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set = load_datasets()
    classes = datasets.CIFAR10(root='./data', train=False).classes
    visualize_samples(train_set, classes)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    model = ConvNetCIFAR10(dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Halve LR every 10 epochs

    train_losses, val_losses, train_accs, val_accs, test_losses, test_accs = [], [], [], [], [], []
    best_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

    torch.save(best_state, "best_cifar10_model.pth")
    model.load_state_dict(torch.load("best_cifar10_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc*100:.2f}%")

    # --- Plotting ---
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot([a * 100 for a in train_accs], label="Train Acc")
    plt.plot([a * 100 for a in val_accs], label="Val Acc")
    plt.plot([a * 100 for a in test_accs], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "acc_curve.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
