import argparse
import json
import os
import sys
import numpy as np

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.metrics import calculate_metrics, print_metrics_report

def main():
    parser = argparse.ArgumentParser(description="Orivis Model Evaluation Script")
    parser.add_argument("--results_file", type=str, help="Path to JSON file containing predictions (y_true, y_scores)")
    parser.add_argument("--output_report", type=str, default="evaluation_report.json", help="Path to save the metrics report")
    parser.add_argument("--dataset_name", type=str, default="Unknown", help="Name of the dataset being evaluated")
    
    args = parser.parse_args()
    
    if args.results_file and os.path.exists(args.results_file):
        with open(args.results_file, 'r') as f:
            data = json.load(f)
            y_true = data.get("y_true", [])
            y_scores = data.get("y_scores", [])
    else:
        print("No results file provided or file not found. Running with synthetic dummy data for demonstration.")
        # Dummy data: 100 samples, 40 real (0), 60 fake (1)
        # Scores are somewhat separated but with noise
        y_true = np.array([0]*40 + [1]*60)
        y_scores = np.concatenate([
            np.random.normal(0.2, 0.15, 40), # Real scores centered at 0.2
            np.random.normal(0.8, 0.2, 60)   # Fake scores centered at 0.8
        ])
        y_scores = np.clip(y_scores, 0, 1)

    if not y_true or not y_scores:
        print("Error: Empty ground truth or scores.")
        return

    metrics = calculate_metrics(y_true, y_scores)
    
    # Print human-readable report
    print_metrics_report(metrics, title=f"Evaluation Report: {args.dataset_name}")
    
    # Save machine-readable report
    with open(args.output_report, 'w') as f:
        json.dump({
            "dataset": args.dataset_name,
            "metrics": metrics
        }, f, indent=4)
    
    print(f"Report saved to {args.output_report}")

if __name__ == "__main__":
    main()
