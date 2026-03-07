import os
import sys
import torch
import json
import argparse
from tqdm import tqdm

                  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.video.video_model import get_xception_model
from src.video.video_loader import create_video_dataloader
from src.utils.metrics import calculate_metrics, print_metrics_report

def parse_test_list(test_list_path, data_dir):
    """
    Parses List_of_testing_videos.txt.
    Returns: List of (video_path, label)
    Note: Test list uses 1 for Real, 0 for Fake. 
    We convert to: 0 for Real, 1 for Fake (Synthesis) to match training.
    """
    data = []
    with open(test_list_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                label_raw = int(parts[0])
                rel_path = parts[1]
                                                  
                label = 0 if label_raw == 1 else 1
                full_path = os.path.join(data_dir, rel_path)
                if os.path.exists(full_path):
                    data.append((full_path, label))
    return data

def run_evaluation(model, test_loader, device):
    model.eval()
    y_true = []
    y_scores = []
    
    print(f"Running inference on {len(test_loader.dataset)} videos...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
                                                                             
            probs = torch.softmax(outputs, dim=1)
            scores = probs[:, 1].cpu().numpy()
            
            y_true.extend(labels.numpy())
            y_scores.extend(scores)
            
    return y_true, y_scores

def main():
    parser = argparse.ArgumentParser(description="Full Evaluation on Celeb-DF v2 Test Split")
    parser.add_argument("--data_dir", type=str, default="Celeb-DF-v2", help="Path to Celeb-DF (v2) root directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames per video")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_json", type=str, default="results/test_metrics.json", help="Path to save metrics")
    
    args = parser.parse_args()
    
    test_list_path = os.path.join(args.data_dir, "List_of_testing_videos.txt")
    if not os.path.exists(test_list_path):
        print(f"Error: {test_list_path} not found.")
        return

                    
    test_data = parse_test_list(test_list_path, args.data_dir)
    if not test_data:
        print("No test data found.")
        return
    
    paths, labels = zip(*test_data)
    test_loader = create_video_dataloader(list(paths), list(labels), 
                                         batch_size=args.batch_size, 
                                         num_frames=args.num_frames, 
                                         shuffle=False)
    
                
    print(f"Loading model from {args.model_path}...")
    model = get_xception_model(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    
          
    y_true, y_scores = run_evaluation(model, test_loader, args.device)
    
                     
    metrics = calculate_metrics(y_true, y_scores)
    print_metrics_report(metrics, title="Celeb-DF v2 Test Split Evaluation")
    
                  
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
