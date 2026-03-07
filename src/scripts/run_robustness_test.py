import os
import sys
import torch
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm

                  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.video.video_model import get_xception_model
from src.video.video_pipeline import VideoPipeline
from src.video.robustness import RobustnessTester
from src.utils.metrics import calculate_metrics

def evaluate_on_transformed_batch(model, video_paths, labels_true, transform_fn, transform_name, device):
    model.eval()
    y_scores = []
    pipeline = VideoPipeline(target_size=(299, 299))
    
    print(f"Testing robustness: {transform_name}...")
    for path in tqdm(video_paths):
                                             
        frames = pipeline.extract_keyframes(path, num_frames=10)
        if not frames:
            y_scores.append(0.5) 
            continue
            
        transformed_frames = []
        for frame in frames:
                                  
            t_frame = transform_fn(frame)
                        
            t_frame = cv2.resize(t_frame, (299, 299))
            t_frame = torch.from_numpy(t_frame).permute(2, 0, 1).float() / 255.0
            transformed_frames.append(t_frame)
            
        batch = torch.stack(transformed_frames).to(device)
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            avg_score = probs[:, 1].mean().item()
            y_scores.append(avg_score)
            
    metrics = calculate_metrics(labels_true, y_scores)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Robustness Testing for Orivis Model")
    parser.add_argument("--data_dir", type=str, default="Celeb-DF-v2", help="Path to Celeb-DF (v2) root directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to test per category")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    args = parser.parse_args()
    
                
    print(f"Loading model from {args.model_path}...")
    model = get_xception_model(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    
                                    
    test_list_path = os.path.join(args.data_dir, "List_of_testing_videos.txt")
    from src.scripts.run_full_eval import parse_test_list
    test_data = parse_test_list(test_list_path, args.data_dir)
    
    real_vids = [d for d in test_data if d[1] == 0]
    fake_vids = [d for d in test_data if d[1] == 1]
    
    sample_data = real_vids[:args.num_samples] + fake_vids[:args.num_samples]
    paths, labels = zip(*sample_data)
    
    tester = RobustnessTester()
    results = {}
    
    results["Baseline"] = evaluate_on_transformed_batch(
        model, paths, labels, lambda x: x, "Baseline", args.device
    )
    
    results["JPEG_Q50"] = evaluate_on_transformed_batch(
        model, paths, labels, lambda x: tester.apply_jpeg_compression(x, 50), "JPEG Compression (Q=50)", args.device
    )
    
    results["GaussianNoise_S10"] = evaluate_on_transformed_batch(
        model, paths, labels, lambda x: tester.add_gaussian_noise(x, sigma=10), "Gaussian Noise (Sigma=10)", args.device
    )
    
    results["Resize_0.5"] = evaluate_on_transformed_batch(
        model, paths, labels, lambda x: tester.apply_resizing(x, 0.5), "Resizing (0.5x)", args.device
    )
    
    print("\n--- Robustness Testing Report ---")
    print(f"{'Transformation':<25} | {'AUC':<10} | {'F1':<10}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:<25} | {m['auc']:<10.4f} | {m['f1']:<10.4f}")
        
    os.makedirs("results", exist_ok=True)
    with open("results/robustness_report.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nReport saved to results/robustness_report.json")

if __name__ == "__main__":
    main()
