import torch
import math
import os
import torch.nn.functional as F
from src.inference.base import InferenceService
from src.video.video_model import XceptionBaseline
from src.video.video_pipeline import VideoPipeline
from src.video.gradcam import GradCAM
import numpy as np
import cv2
from typing import Dict, Any, List, Optional

class VideoInferenceService(InferenceService):
    """
    Concrete implementation of InferenceService for video deepfake detection.
    """
    def __init__(self, model_path: str = "models/video_baseline.pth", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        # Xception requires 299x299 input size
        self.pipeline = VideoPipeline(target_size=(299, 299))
        self.gradcam = None
        # Calibrated threshold based on evaluation (EER threshold)
        self.threshold = 0.912
        super().__init__(model_id="video_xception_v1", device=device)

    def load_model(self):
        self.model = XceptionBaseline(num_classes=2, pretrained=False)
        
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded video model from {self.model_path}")
        else:
            print(f"Warning: Video model checkpoint {self.model_path} not found.")
            
        self.model.to(self.device)
        self.model.eval()
        
        for module in self.model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False
        
        # Initialize Grad-CAM with target layer (block12 for Xception is a good choice)
        try:
            target_layer = self.model.backbone.block12.rep[-1] 
            self.gradcam = GradCAM(self.model, target_layer)
        except Exception as e:
            print(f"Warning: Could not initialize Grad-CAM: {e}")

    def preprocess(self, input_data: str) -> torch.Tensor:
        video_tensor = self.pipeline.process_video(input_data)
        return video_tensor.to(self.device)

    def predict(self, preprocessed_data: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass.
        Input: (B, N, C, H, W)
        """
        if len(preprocessed_data.shape) == 4:
            preprocessed_data = preprocessed_data.unsqueeze(0)
            
        with torch.no_grad():
            logits = self.model(preprocessed_data)
        return logits

    def generate_heatmap(self, preprocessed_data: torch.Tensor, original_img: np.ndarray) -> np.ndarray:
        """
        Generate overlayed heatmap on original image.
        """
        if self.gradcam is None:
            return original_img
            
        heatmap = self.gradcam.generate_heatmap(preprocessed_data)
        overlayed = self.gradcam.visualize_on_image(original_img, heatmap)
        return overlayed

    def calibrate_probability(self, prob: float) -> float:
        """
        Pushes probabilities away from the threshold (0.912) to be more decisive.
        Uses a sigmoid function with high gain to ensure scores around the threshold
        jump to high/low confidence regions (~80-90%).
        """
        # Gain factor: higher = steeper jump from real to fake
        gain = 500.0
        diff = prob - self.threshold
        
        # Sigmoid function: 1 / (1 + exp(-gain * diff))
        try:
            calibrated = 1 / (1 + math.exp(-gain * diff))
        except OverflowError:
            calibrated = 1.0 if diff > 0 else 0.0
            
        return calibrated

    def postprocess(self, raw_results: torch.Tensor) -> Dict[str, Any]:
        """
        Convert raw results into calibrated format.
        Index 1 is the synthetic/fake class probability.
        """
        probs = F.softmax(raw_results, dim=1)
        fake_prob_raw = probs[0][1].item() 
        
        # Calibrate so 0.5 is the classification line
        fake_prob = self.calibrate_probability(fake_prob_raw)
        
        print(f"DEBUG: Video Detection - Raw: {fake_prob_raw:.4f}, Calibrated: {fake_prob:.4f}, Threshold: {self.threshold}")
        
        return {
            "probability": fake_prob,
            "label": "fake" if fake_prob >= 0.5 else "real",
            "confidence": float(max(probs[0]).item()),
            "metadata": {
                "raw_probability": fake_prob_raw,
                "threshold": self.threshold
            }
        }
