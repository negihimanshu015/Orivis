class FusionModel:
    """
    Weighted fusion of multimodal predictions.
    Combines video and audio fake probabilities into a single score.
    """
    def __init__(self, video_weight: float = 0.6, audio_weight: float = 0.4):
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        
                                                  
        total_weight = video_weight + audio_weight
        if total_weight != 1.0:
            self.video_weight /= total_weight
            self.audio_weight /= total_weight

    def predict(self, video_fake_prob: float, audio_fake_prob: float) -> float:
        """
        Apply weighted averaging fusion.
        """
        final_score = (self.video_weight * video_fake_prob) + (self.audio_weight * audio_fake_prob)
        return final_score

    def get_weighted_prediction(self, video_prob: float, audio_prob: float) -> dict:
        """
        Returns structured results for backend integration.
        """
        final_prob = self.predict(video_prob, audio_prob)
        return {
            "video_probability": float(video_prob),
            "audio_probability": float(audio_prob),
            "final_synthetic_probability": float(final_prob),
            "label": "fake" if final_prob > 0.5 else "real"
        }
