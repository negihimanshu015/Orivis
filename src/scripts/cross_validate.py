import os
import sys
import torch
import json

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.video.xception import get_xception_model
from src.models.video.efficientnet import get_efficientnet_model
from src.scripts.train_utils import run_training, validate
from src.utils.video_loader import create_video_dataloader

def cross_dataset_eval(model_path, model_type, test_dataset_name, video_paths, labels, device="cpu"):
    """
    Load a trained model and evaluate it on an entirely different dataset.
    """
    if model_type == "xception":
        model = get_xception_model(pretrained=False)
    else:
        model = get_efficientnet_model(pretrained=False)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    test_loader = create_video_dataloader(video_paths, labels, batch_size=4, shuffle=False)
    
    print(f"\nStarting Cross-Dataset Evaluation on: {test_dataset_name}")
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    report = {
        "source_model": model_path,
        "test_dataset": test_dataset_name,
        "metrics": test_metrics
    }
    
    print(f"Cross-Dataset AUC: {test_metrics['auc']:.4f}")
    return report

if __name__ == "__main__":
    # Example usage (placeholders)
    # cross_dataset_eval("models/xception_ff.pth", "xception", "DFDC", [...], [...])
    pass
