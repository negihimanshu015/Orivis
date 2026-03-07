import torch
import os
from typing import Dict, Any

from src.fusion.fusion_model import FusionModel
from src.inference.video_service import VideoInferenceService
from src.inference.audio_service import AudioInferenceService

class FusionPipeline:
    """
    Multimodal fusion pipeline that runs video and audio detectors
    and combines their outputs using the actual models in the models folder.
    """
    def __init__(
        self, 
        video_model_path: str = "models/video_baseline.pth", 
        audio_model_path: str = "models/audio_baseline.pth",
        video_weight: float = 0.6, 
        audio_weight: float = 0.4,
        device: str = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
                                                                     
        self.video_service = VideoInferenceService(model_path=video_model_path, device=device)
            
        self.audio_service = AudioInferenceService(model_path=audio_model_path, device=device)
        self.fusion = FusionModel(video_weight=video_weight, audio_weight=audio_weight)

    def run_fusion(self, video_path: str, audio_path: str = None) -> Dict[str, Any]:
        """
        Run both models and fuse results.
        If audio_path is None, it uses video_path (extracting audio if possible).
        """
                                
        v_res = self.video_service.run_inference(video_path)
        video_prob = v_res.get("probability", 0.5)
            
                                
        audio_target = audio_path if audio_path else video_path
        a_res = self.audio_service.run_inference(audio_target)
        audio_prob = a_res.get("probability", 0.5)
            
                 
        results = self.fusion.get_weighted_prediction(video_prob, audio_prob)
        
        return results
