import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import Precision, F1Score, Recall, JaccardIndex

device = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_accuracy(output, target):
    _, preds = torch.max(output, 1)
    corrects = torch.sum(preds == target).item()
    accuracy = corrects / (target.size(0) * target.size(1) * target.size(2))
    return accuracy

def calculate_precision(output, mask, num_classes, device):
    precision = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    preds = torch.argmax(output, dim=1)
    return precision(preds, mask)

def calculate_f1(output, mask, num_classes, device):
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    preds = torch.argmax(output, dim=1)
    return f1(preds, mask)

def calculate_recall(output, mask, num_classes, device):
    recall = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    preds = torch.argmax(output, dim=1)
    return recall(preds, mask)

def calculate_iou(output, mask, num_classes, device):
    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    preds = torch.argmax(output, dim=1)
    return jaccard(preds, mask)

import torch
from tqdm import tqdm


def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_precision = 0.0
        val_f1 = 0.0
        val_recall = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(outputs, labels)
                val_precision += calculate_precision(outputs, labels, num_classes, device)
                val_f1 += calculate_f1(outputs, labels, num_classes, device)
                val_recall += calculate_recall(outputs, labels, num_classes, device)
                val_iou += calculate_iou(outputs, labels, num_classes, device)
        
        # Compute average metrics
        num_batches = len(val_loader)
        val_loss /= num_batches
        val_accuracy /= num_batches
        val_precision /= num_batches
        val_f1 /= num_batches
        val_recall /= num_batches
        val_iou /= num_batches
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val Precision: {val_precision:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}, Val IoU: {val_iou:.4f}")
        
    print("Training completed.")
