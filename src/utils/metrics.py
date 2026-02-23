import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calculate_metrics(y_true, y_scores, threshold=0.5):
    """
    Calculate core evaluation metrics for synthetic media detection.
    
    Args:
        y_true (array-like): Ground truth labels (0 for real, 1 for fake).
        y_scores (array-like): Predicted probability scores for the 'fake' class.
        threshold (float): Classification threshold for F1 and other discrete metrics.
        
    Returns:
        dict: A dictionary containing AUC, F1, EER, Precision, and Recall.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    
    # Standard metrics
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Equal Error Rate (EER)
    # EER is where FPR == FNR (1 - TPR)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    
    return {
        "auc": float(roc_auc),
        "f1": float(f1),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "precision": float(precision),
        "recall": float(recall)
    }

def print_metrics_report(metrics, title="Evaluation Report"):
    """
    Print a formatted report of the metrics.
    """
    print(f"\n{'='*30}")
    print(f" {title} ")
    print(f"{'='*30}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"EER:       {metrics['eer']:.4f} (at threshold {metrics['eer_threshold']:.4f})")
    print(f"F1-Score:  {metrics['f1']:.4f} (at threshold 0.5000)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"{'='*30}\n")
