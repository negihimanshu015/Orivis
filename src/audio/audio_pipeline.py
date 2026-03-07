import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Optional, Tuple
import os

class AudioPipeline:
    """
    Utilities for audio processing, including loading, fixed-length padding/trimming,
    and log-mel spectrogram conversion using torchaudio.
    """
    def __init__(
        self, 
        sample_rate: int = 16000, 
        duration: int = 4, 
        n_mels: int = 128,
        n_fft: int = 400,
        hop_length: int = 160
    ):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = sample_rate * duration
        self.n_mels = n_mels
        
                                  
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and resample if necessary.
        Uses moviepy for robust extraction from videos and librosa/soundfile for audio.
        """
        ext = os.path.splitext(audio_path)[1].lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
        
        if is_video:
            from moviepy import VideoFileClip
            import tempfile
            
                                                   
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp_path = tmp.name
                
            try:
                with VideoFileClip(audio_path) as clip:
                    if clip.audio is None:
                                                                    
                        waveform = np.zeros(self.target_samples)
                        sr = self.sample_rate
                    else:
                        clip.audio.write_audiofile(tmp_path, fps=self.sample_rate, verbose=False, logger=None)
                        waveform, sr = librosa.load(tmp_path, sr=self.sample_rate)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except PermissionError:
                        print(f"Warning: Could not remove temporary file {tmp_path}")
        else:
                                                 
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            
        waveform = torch.from_numpy(waveform).unsqueeze(0).to(torch.float32)               
            
        return waveform

    def pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Ensure waveform is exactly target_samples long.
        """
        num_samples = waveform.shape[1]
        
        if num_samples > self.target_samples:
                  
            waveform = waveform[:, :self.target_samples]
        elif num_samples < self.target_samples:
                            
            padding = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform

    def to_log_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to log-mel spectrogram.
        Returns shape (1, n_mels, time_frames).
        """
        mel_spec = self.mel_spectrogram(waveform)
        log_mel = self.amplitude_to_db(mel_spec)
        
                                                       
        mean = log_mel.mean()
        std = log_mel.std()
        if std > 0:
            log_mel = (log_mel - mean) / std
            
        return log_mel

    def process_audio(self, audio_path: str) -> torch.Tensor:
        """
        Full pipeline: Load -> Pad/Trim -> Log-Mel -> Normalize.
        """
        waveform = self.load_audio(audio_path)
        waveform = self.pad_or_trim(waveform)
        log_mel = self.to_log_mel(waveform)
        return log_mel
