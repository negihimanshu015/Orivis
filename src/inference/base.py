from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time

class InferenceService(ABC):
    """
    Abstract Base Class for inference services.
    Provides a consistent interface for different model backbones.
    """
    
    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model into memory/GPU."""
        pass

    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Preprocess raw input (video path, audio bytes, etc.) for the model."""
        pass

    @abstractmethod
    def predict(self, preprocessed_data: Any) -> Dict[str, Any]:
        """Perform inference and return raw results."""
        pass

    @abstractmethod
    def postprocess(self, raw_results: Any) -> Dict[str, Any]:
        """Convert raw results into standard Orivis format (probability, confidence)."""
        pass

    def run_inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Full inference pipeline: preprocess -> predict -> postprocess.
        Tracks execution time.
        """
        start_time = time.time()
        
        preprocessed = self.preprocess(input_data)
        raw_results = self.predict(preprocessed)
        final_results = self.postprocess(raw_results)
        
        execution_time = time.time() - start_time
        final_results["metadata"] = {
            "model_id": self.model_id,
            "device": self.device,
            "execution_time_sec": execution_time
        }
        
        return final_results
