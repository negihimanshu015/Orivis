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
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_keyframes(self, video_path: str, num_frames: int = 10) -> List[np.ndarray]:
        """
        Extract a fixed number of key-frames at regular intervals from a video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return []

        # Calculate indices for frames at regular intervals
        interval = max(1, total_frames // num_frames)
        frame_indices = [i * interval for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        
        cap.release()
        
        # If we didn't get enough frames, pad with the last one or black frames
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
