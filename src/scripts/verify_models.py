import torch
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.video.xception import get_xception_model
from src.models.video.efficientnet import get_efficientnet_model

def verify_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verifying models on {device}...")
    
    # Dummy input: (Batch=2, NumFrames=5, C=3, H=224, W=224)
    dummy_input = torch.randn(2, 5, 3, 224, 224).to(device)
    
    # 1. Xception
    print("Testing Xception...")
    xception = get_xception_model(pretrained=False).to(device)
    out_xc = xception(dummy_input)
    print(f"Xception output shape: {out_xc.shape}")
    assert out_xc.shape == (2, 2)
    
    # 2. EfficientNet
    print("Testing EfficientNet...")
    effnet = get_efficientnet_model(pretrained=False).to(device)
    out_eff = effnet(dummy_input)
    print(f"EfficientNet output shape: {out_eff.shape}")
    assert out_eff.shape == (2, 2)
    
    print("Model verification successful!")

if __name__ == "__main__":
    verify_models()
