import os
import sys
import argparse
import random
import json
from datetime import datetime
import torch

                  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.video.video_model import get_xception_model
from src.video.efficientnet import get_efficientnet_model
from src.scripts.train_utils import run_training
from src.video.video_loader import create_video_dataloader

def parse_celebdf_v2(data_dir):
    """
    Parses Celeb-DF (v2) directory structure recursivly.
    Returns: List of (video_path, label)
    Labels: 0 for Real, 1 for Synthesis
    """
    real_dir = os.path.join(data_dir, 'Celeb-real')
    synthesis_dir = os.path.join(data_dir, 'Celeb-synthesis')
    youtube_dir = os.path.join(data_dir, 'YouTube-real')
    
    data = []
    
                 
    for d in [real_dir, youtube_dir]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith('.mp4'):
                        data.append((os.path.join(root, f), 0))
    
                      
    if os.path.exists(synthesis_dir):
        for root, _, files in os.walk(synthesis_dir):
            for f in files:
                if f.lower().endswith('.mp4'):
                    data.append((os.path.join(root, f), 1))
                
    return data

def main():
    parser = argparse.ArgumentParser(description="Orivis Training Script for Celeb-DF (v2)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Celeb-DF (v2) root directory")
    parser.add_argument("--model_type", type=str, default="xception", choices=["xception", "efficientnet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames to extract per video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save models and logs")
    
    args = parser.parse_args()
    
                             
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model_type}_{timestamp}"
    save_path = os.path.join(args.output_dir, f"{run_name}_best.pth")
    
               
    print(f"Loading Celeb-DF (v2) data from: {args.data_dir}")
    all_data = parse_celebdf_v2(args.data_dir)
    random.shuffle(all_data)
    
    if not all_data:
        print("Error: No data found in the specified directory.")
        return

                
    split_idx = int(len(all_data) * (1 - args.val_split))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    
    print(f"Total samples: {len(all_data)} (Train: {len(train_data)}, Val: {len(val_data)})")
    
                 
    train_loader = create_video_dataloader(list(train_paths), list(train_labels), 
                                          batch_size=args.batch_size, num_frames=args.num_frames)
    val_loader = create_video_dataloader(list(val_paths), list(val_labels), 
                                        batch_size=args.batch_size, num_frames=args.num_frames, shuffle=False)
    
           
    if args.model_type == "xception":
        model = get_xception_model(pretrained=True)
    else:
        model = get_efficientnet_model(pretrained=True)
        
                    
    print(f"Starting training on {args.device}...")
    best_auc = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        save_path=save_path
    )
    
    print(f"Training completed. Best Val AUC: {best_auc:.4f}")
    print(f"Best model saved to: {save_path}")

if __name__ == "__main__":
    main()
