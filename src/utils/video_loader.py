import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from src.utils.video_pipeline import VideoPipeline

class OrivisVideoDataset(Dataset):
    """
    Standard PyTorch Dataset for Orivis video detection.
    Expects a list of video paths and corresponding labels.
    """
    def __init__(self, video_paths: List[str], labels: List[int], num_frames: int = 10, target_size=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.pipeline = VideoPipeline(target_size=target_size)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Process video to get (num_frames, C, H, W)
            video_tensor = self.pipeline.process_video(video_path, num_frames=self.num_frames)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return zero tensor on error
            video_tensor = torch.zeros((self.num_frames, 3, 224, 224))
            
        return video_tensor, torch.tensor(label, dtype=torch.long)

def create_video_dataloader(video_paths: List[str], labels: List[int], batch_size: int = 4, shuffle: bool = True, **kwargs):
    """
    Helper to create a DataLoader for video data.
    """
    dataset = OrivisVideoDataset(video_paths, labels, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
