import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
from typing import Optional

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.metrics import calculate_metrics, print_metrics_report
from src.utils.video_loader import create_video_dataloader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_y_true = []
    all_y_scores = []
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Collect for metrics (assuming binary classification with softmax outputs)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]
        all_y_true.extend(labels.cpu().numpy())
        all_y_scores.extend(probabilities.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_y_true, all_y_scores)
    return epoch_loss, metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_y_true = []
    all_y_scores = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            all_y_true.extend(labels.cpu().numpy())
            all_y_scores.extend(probabilities.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_y_true, all_y_scores)
    return val_loss, metrics

def run_training(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4, device="cpu", save_path="best_model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    best_val_auc = 0.0
    
    for epoch in range(num_epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_metrics['auc']:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val AUC:   {val_metrics['auc']:.4f}")
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model with AUC: {best_val_auc:.4f}")
            
    return best_val_auc
