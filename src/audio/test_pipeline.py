import torch
import os
import numpy as np
import scipy.io.wavfile as wav
from src.audio.audio_pipeline import AudioPipeline
from src.audio.audio_model import SimpleAudioCNN
from src.audio.audio_loader import ASVspoofDataset
from torch.utils.data import DataLoader

def create_dummy_wav(path, duration=4, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    samples = np.random.uniform(-1, 1, sr * duration).astype(np.float32)
    wav.write(path, sr, samples)

def test_pipeline():
    print("Testing AudioPipeline...")
    pipeline = AudioPipeline()
    dummy_wav = "tmp/dummy.wav"
    create_dummy_wav(dummy_wav)
    
    log_mel = pipeline.process_audio(dummy_wav)
    print(f"Log-mel shape: {log_mel.shape}")
    assert log_mel.shape == (1, 128, 401), f"Expected (1, 128, 401), got {log_mel.shape}"
    
    print("Testing SimpleAudioCNN...")
    model = SimpleAudioCNN()
    logits = model(log_mel.unsqueeze(0))
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1,), f"Expected shape (1,), got {logits.shape}"
    
    print("Testing ASVspoofDataset (Dummy)...")
    dummy_protocol = "tmp/dummy_protocol.txt"
    with open(dummy_protocol, "w") as f:
        f.write("S001 dummy - - bonafide\n")
        f.write("S002 dummy2 - - spoof\n")
    
    create_dummy_wav("tmp/dummy2.wav")
    
    dataset = ASVspoofDataset(root_dir="tmp", protocol_file=dummy_protocol, pipeline=pipeline)
    print(f"Dataset length: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=2)
    for batch_mel, batch_labels in loader:
        print(f"Batch mel shape: {batch_mel.shape}")
        print(f"Batch labels: {batch_labels}")
        assert batch_mel.shape == (2, 1, 128, 401)
        break
        
    print("All tests passed!")

if __name__ == "__main__":
    test_pipeline()
