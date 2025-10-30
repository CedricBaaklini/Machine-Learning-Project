import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

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