import cv2
import numpy as np
import os
from typing import List, Optional

class RobustnessTester:
    """
    Utilities for testing model robustness against common media transformations.
    """
    
    @staticmethod
    def apply_h264_compression(video_path: str, output_path: str, crf: int = 23):
        """
        Compress video using H.264 with specific CRF (Constant Rate Factor).
        Higher CRF means more compression/lower quality.
        """
        # Requires ffmpeg installed on system. If not available, this will fail.
        cmd = f"ffmpeg -i {video_path} -vcodec libx264 -crf {crf} {output_path} -y"
        os.system(cmd)

    @staticmethod
    def apply_jpeg_compression(image: np.ndarray, quality: int = 50) -> np.ndarray:
        """
        Apply JPEG compression to an image.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, 1)

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 10) -> np.ndarray:
        """
        Add Gaussian noise to an image.
        """
        gauss = np.random.normal(mean, sigma, image.shape).astype('float32')
        noisy = image.astype('float32') + gauss
        return np.clip(noisy, 0, 255).astype('uint8')

    @staticmethod
    def apply_resizing(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """
        Degrade image by downsampling and then upsampling back to original size.
        """
        h, w = image.shape[:2]
        low_res = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
