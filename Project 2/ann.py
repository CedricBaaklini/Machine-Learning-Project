import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

torch.manual_seed(42)

batch_size = 128
lr = 1e-3
num_epochs = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(inplace=True), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 10))
        
    def forward(self, x):
        return self.net(x)

model = SimpleANN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(loader):
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

if __name__ == "__main__":
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_size = 50_000
    val_size = 10_000
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model.to(device)

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
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}%")

    test_loss, test_acc = evaluate(test_loader)
    print(f"Test: loss={test_loss:.4f} acc={test_acc * 100:.2f}%")