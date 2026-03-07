import torch
import sys
import os
from src.audio.audio_model import SimpleAudioCNN
from src.audio.audio_pipeline import AudioPipeline

def infer(audio_path, model_path="models/audio_baseline.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
                                     
    pipeline = AudioPipeline()
    try:
        log_mel = pipeline.process_audio(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
        
                                          
    log_mel = log_mel.unsqueeze(0).to(device)
    
                   
    model = SimpleAudioCNN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model checkpoint {model_path} not found. Using uninitialized model.")
    
    model.eval()
    
                
    with torch.no_grad():
        logits = model(log_mel)
        probability = torch.sigmoid(logits).item()
        
    prediction = "Spoof" if probability >= 0.5 else "Real"
    
    print(f"\nAudio: {os.path.basename(audio_path)}")
    print(f"Synthetic Probability: {probability:.4f}")
    print(f"Prediction: {prediction}")
    
    return probability, prediction

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/audio/infer_audio.py <audio_path> [model_path]")
    else:
        path = sys.argv[1]
        m_path = sys.argv[2] if len(sys.argv) > 2 else "models/audio_baseline.pth"
        infer(path, m_path)
