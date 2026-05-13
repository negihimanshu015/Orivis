import cv2
import os
import numpy as np
from PIL import Image
from typing import List, Optional
import torch
from torchvision import transforms

class VideoPipeline:
    """
    Utilities for video processing, including key-frame extraction and preprocessing.
    """
    def __init__(self, target_size=(299, 299)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_keyframes(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """
        Extract a fixed number of key-frames at regular intervals from a video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                return []

            interval = max(1, total_frames // num_frames)
            frame_indices = [i * interval for i in range(num_frames)]

            # Sequential read is more robust than cap.set() across different OS backends
            frames = []
            frame_idx = 0
            for target_idx in frame_indices:
                while frame_idx < target_idx:
                    cap.read()  # Skip frames
                    frame_idx += 1
                
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    frame_idx += 1
                else:
                    break
        finally:
            cap.release()
        
                                                                               
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((*self.target_size, 3), dtype=np.uint8))
                
        return frames[:num_frames]

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Apply resizing, normalization, and conversion to tensor.
        """
        image = Image.fromarray(frame)
        return self.transform(image)

    def process_video(self, video_path: str, num_frames: int = 10) -> torch.Tensor:
        """
        Extract and preprocess frames, returning a batch tensor (N, C, H, W).
        """
        frames = self.extract_keyframes(video_path, num_frames)
        processed_frames = [self.preprocess_frame(f) for f in frames]
        return torch.stack(processed_frames)
