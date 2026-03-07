import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import argparse

from src.audio.audio_loader import create_audio_dataloader
from src.audio.audio_model import SimpleAudioCNN
from src.utils.metrics import calculate_metrics, print_metrics_report

def train(args):
                   
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
           
    root_dir = args.train_audio_dir
    protocol_file = args.train_protocol
    val_root_dir = args.val_audio_dir
    val_protocol_file = args.val_protocol
    
                              
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
                          
    if not os.path.exists(root_dir) or not os.path.exists(protocol_file):
        print(f"Error: Training data not found at {root_dir} or {protocol_file}")
        print("Please download the ASVspoof 2019 LA dataset and provide the correct paths.")
        return

             
    print("Loading datasets...")
    train_loader = create_audio_dataloader(root_dir, protocol_file, batch_size=batch_size)
    val_loader = create_audio_dataloader(val_root_dir, val_protocol_file, batch_size=batch_size, shuffle=False)
    
                            
    model = SimpleAudioCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for waveforms, labels in pbar:
            waveforms, labels = waveforms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
                    
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}")
            for waveforms, labels in val_pbar:
                waveforms, labels = waveforms.to(device), labels.to(device)
                logits = model(waveforms)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                                   
                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        metrics = calculate_metrics(all_labels, all_scores)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print_metrics_report(metrics)
        
                         
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save(model.state_dict(), "models/audio_baseline.pth")
            
                                  
            results = {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                **metrics
            }
            with open("experiments/audio_baseline.yaml", "w") as f:
                yaml.dump(results, f)
                
    print("Training complete. Best AUC:", best_auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Deepfake Detector")
    parser.add_argument("--train_audio_dir", type=str, required=True, help="Path to training flac files")
    parser.add_argument("--train_protocol", type=str, required=True, help="Path to training protocol file")
    parser.add_argument("--val_audio_dir", type=str, required=True, help="Path to validation flac files")
    parser.add_argument("--val_protocol", type=str, required=True, help="Path to validation protocol file")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    train(args)
