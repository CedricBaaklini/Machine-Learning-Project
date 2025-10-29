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
            
    avg_loss = loss_sum / max(1, batches)
    acc = correct / max(1, batches)
    return avg_loss, acc
    #return loss_sum / max(1, batches), correct / max(1, total)