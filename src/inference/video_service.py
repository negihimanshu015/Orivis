import torch
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
        self.pipeline = VideoPipeline(target_size=(224, 224))
        self.gradcam = None
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
        Input: (N, C, H, W).
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

    def postprocess(self, raw_results: torch.Tensor) -> Dict[str, Any]:
        probs = F.softmax(raw_results, dim=1)
        fake_prob = probs[0][1].item()
        
        return {
            "probability": fake_prob,
            "label": "fake" if fake_prob > 0.5 else "real",
            "confidence": float(max(probs[0]).item())
        }
