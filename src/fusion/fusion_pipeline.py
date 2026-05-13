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

    def run_fusion(self, video_path: str, audio_path: str = None, generate_heatmaps: bool = True) -> Dict[str, Any]:
        """
        Run both models and fuse results.
        If audio_path is None, it uses video_path (extracting audio if possible).
        """
        # Run Video Inference
        v_res = self.video_service.run_inference(video_path)
        video_prob = v_res.get("probability", 0.5)
        
        # Run Audio Inference
        audio_target = audio_path if audio_path else video_path
        a_res = self.audio_service.run_inference(audio_target)
        audio_prob = a_res.get("probability", 0.5)
        
        # Dynamic Fusion: Bypass if audio is missing/silent
        if a_res.get("metadata", {}).get("is_silent"):
            results = {
                "video_probability": float(video_prob),
                "audio_probability": float(audio_prob),
                "final_synthetic_probability": float(video_prob),
                "label": "fake" if video_prob > 0.5 else "real"
            }
        else:
            # Weighted Fusion
            results = self.fusion.get_weighted_prediction(video_prob, audio_prob)
        
        # Add detailed results
        results["video_details"] = v_res
        results["audio_details"] = a_res
        
        # Generate heatmap if requested and video detected suspicious
        if generate_heatmaps and video_prob > 0.3:
            try:
                # Process video to get frames and tensors
                video_tensor = self.video_service.preprocess(video_path)
                # Take the middle frame for Grad-CAM sample
                mid_idx = video_tensor.shape[1] // 2 if len(video_tensor.shape) == 5 else 0
                sample_tensor = video_tensor[0, mid_idx].unsqueeze(0) if len(video_tensor.shape) == 5 else video_tensor[0].unsqueeze(0)
                
                # Get the actual image frame
                from src.video.video_pipeline import VideoPipeline
                pipeline = VideoPipeline()
                frames = pipeline.extract_keyframes(video_path, num_frames=10)
                sample_frame = frames[mid_idx]
                
                overlayed = self.video_service.generate_heatmap(sample_tensor, sample_frame)
                results["heatmap"] = overlayed
            except Exception as e:
                print(f"Error generating heatmap in pipeline: {e}")
                results["heatmap"] = None
        
        return results
