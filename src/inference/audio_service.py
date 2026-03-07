import torch
import os
from src.inference.base import InferenceService
from src.audio.audio_model import SimpleAudioCNN
from src.audio.audio_pipeline import AudioPipeline
from typing import Dict, Any

class AudioInferenceService(InferenceService):
    """
    Concrete implementation of InferenceService for audio deepfake detection.
    """
    def __init__(self, model_path: str = "models/audio_baseline.pth", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.pipeline = AudioPipeline()
        super().__init__(model_id="audio_cnn_v1", device=device)

    def load_model(self):
                                       
        self.model = SimpleAudioCNN(num_classes=1)
        
                      
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded audio model from {self.model_path}")
        else:
            print(f"Warning: Audio model checkpoint {self.model_path} not found.")
            
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, input_data: str) -> torch.Tensor:
        """
        Input: path to audio file.
        Output: log-mel spectrogram (1, n_mels, T).
        """
        log_mel = self.pipeline.process_audio(input_data)
        return log_mel.to(self.device)

    def predict(self, preprocessed_data: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass.
        """
                                                                                        
        self._is_silent = (torch.std(preprocessed_data) == 0).item()
        
                                              
        preprocessed_data = preprocessed_data.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(preprocessed_data)
        return logits

    def postprocess(self, raw_results: torch.Tensor) -> Dict[str, Any]:
        """
        Convert logits to probability.
        """
        if getattr(self, '_is_silent', False):
            fake_prob = 0.0
        else:
            fake_prob = torch.sigmoid(raw_results).item()
        
        return {
            "probability": fake_prob,
            "label": "fake" if fake_prob > 0.5 else "real",
            "confidence": float(fake_prob if fake_prob > 0.5 else 1 - fake_prob)
        }
