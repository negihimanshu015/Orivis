import os
import sys
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm

                  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.video.video_model import get_xception_model
from src.utils.gradcam import GradCAM
from src.video.video_pipeline import VideoPipeline

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization for Orivis Model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a video file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--output_dir", type=str, default="results/gradcam", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
                
    print(f"Loading model from {args.model_path}...")
    model = get_xception_model(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()
    
                                                       
    try:
        target_layer = model.backbone.conv4
    except AttributeError:
        print("Error: Could not find model.backbone.conv4.")
        return
    
    gcam = GradCAM(model, target_layer)
    
                                        
    pipeline = VideoPipeline(target_size=(299, 299))
    frames = pipeline.extract_keyframes(args.video_path, args.num_frames)
    
    if not frames:
        print("Failed to extract frames.")
        return
    
    video_name = os.path.basename(args.video_path).split('.')[0]
    
    for i, frame in enumerate(frames):
                          
        img = cv2.resize(frame, (299, 299))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(args.device)
        
                          
        heatmap = gcam.generate_heatmap(img_tensor)
        
                   
        result = gcam.visualize_on_image(img, heatmap)
        
              
        out_path = os.path.join(args.output_dir, f"{video_name}_frame_{i}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {out_path}")
    
    gcam.remove_hooks()

if __name__ == "__main__":
    main()
