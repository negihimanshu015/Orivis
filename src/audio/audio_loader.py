import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from src.audio.audio_pipeline import AudioPipeline

class ASVspoofDataset(Dataset):
    """
    Dataset for ASVspoof Logical Access data.
    """
    def __init__(
        self, 
        root_dir: str, 
        protocol_file: str, 
        pipeline: Optional[AudioPipeline] = None,
        is_train: bool = True
    ):
        self.root_dir = root_dir
        self.pipeline = pipeline or AudioPipeline()
        self.samples = []
        
        if not os.path.exists(protocol_file):
            print(f"Warning: Protocol file {protocol_file} not found.")
            return

        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                                              
                                                                             
                audio_id = parts[1]
                key = parts[4]
                
                audio_path = os.path.join(root_dir, f"{audio_id}.flac")
                if not os.path.exists(audio_path):
                                               
                    audio_path = os.path.join(root_dir, f"{audio_id}.wav")
                    
                label = 1 if key == "spoof" else 0
                self.samples.append((audio_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path, label = self.samples[idx]
        
        try:
            log_mel = self.pipeline.process_audio(audio_path)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
                                         
            log_mel = torch.zeros((1, self.pipeline.n_mels, 401))                               
            
        return log_mel, torch.tensor(label, dtype=torch.float32)

def create_audio_dataloader(
    root_dir: str, 
    protocol_file: str, 
    batch_size: int = 16, 
    shuffle: bool = True,
    pipeline: Optional[AudioPipeline] = None
):
    dataset = ASVspoofDataset(root_dir, protocol_file, pipeline=pipeline)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
