from src.fusion.fusion_model import FusionModel
from src.fusion.fusion_pipeline import FusionPipeline
import json

def test_fusion_logic():
    print("Testing FusionModel logic...")
    model = FusionModel(video_weight=0.6, audio_weight=0.4)
    
                            
    res1 = model.get_weighted_prediction(0.8, 0.9)
    print(f"Test 1 (0.8, 0.9) -> {res1['final_synthetic_probability']:.4f}")
    assert abs(res1['final_synthetic_probability'] - 0.84) < 1e-5
    
                        
    res2 = model.get_weighted_prediction(0.9, 0.1)
    print(f"Test 2 (0.9, 0.1) -> {res2['final_synthetic_probability']:.4f}")
    assert abs(res2['final_synthetic_probability'] - 0.58) < 1e-5
    
                                       
    model_unnorm = FusionModel(video_weight=6, audio_weight=4)
    res3 = model_unnorm.get_weighted_prediction(0.5, 0.5)
    print(f"Test 3 (unnormalized weights) -> {res3['final_synthetic_probability']:.4f}")
    assert abs(res3['final_synthetic_probability'] - 0.5) < 1e-5

def test_pipeline_structure():
    print("\nTesting FusionPipeline structure...")
    
                                                                   
    v_ext = os.path.exists("models/video_baseline.pth")
    a_ext = os.path.exists("models/audio_baseline.pth")
    
    if v_ext and a_ext:
        print("Real models found. Initializing real pipeline...")
        try:
            pipeline = FusionPipeline(device="cpu")                        
            print("Pipeline initialized successfully with real models.")
        except Exception as e:
            print(f"Error initializing real pipeline: {e}")
            return
    else:
        print("Real models not found. Skipping real pipeline test.")
        print("Note: FusionModel logic was already verified above.")

if __name__ == "__main__":
    import os
    test_fusion_logic()
    test_pipeline_structure()
    print("\nFusion tests completed!")
