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
EPOCHS = 80
LEARNING_RATE = 0.1 #updated
DROPOUT = 0.3
VAL_SPLIT = 0.1
RANDOM_SEED = 42

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)  # Update input size for fc1
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))   # conv1 --> batchnorm --> ReLu
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)               # 32 --> 16
        x = self.relu(self.batchnorm3(self.conv3(x)))   # conv3 --> batchnorm --> ReLu
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.pool(x)               # 16 --> 8
        x = self.relu(self.batchnorm5(self.conv5(x)))   # conv5 --> batchnorm --> ReLu
        x = self.relu(self.batchnorm6(self.conv6(x)))
        x = self.pool(x)               # 8 --> 4
        x = x.view(x.size(0), -1)      # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
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
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.2)  # Drop LR at epochs 40 and 60 by factor 0.2

    train_losses, val_losses, train_accs, val_accs, test_losses, test_accs = [], [], [], [], [], []
    best_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
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
