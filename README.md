# Orivis: Deepfake Detection System

Orivis is a robust face forgery detection system designed to identify manipulated videos using state-of-the-art deep learning architectures.

## Project Structure
- `src/api`: FastAPI backend and endpoint definitions.
- `src/audio`: Audio models, pipelines, and training logic.
- `src/video`: Video models, pipelines, and evaluation utilities.
- `src/fusion`: Multimodal fusion logic and pipeline.
- `src/inference`: Inference services for audio and video.
- `src/scripts`: Training, evaluation, and visualization scripts.
- `src/utils`: Shared helper utilities (metrics).

## Performance Results (Celeb-DF v2)
The baseline Xception model achieves the following performance on the Celeb-DF v2 test split:
- **AUC**: 0.9250
- **EER**: 0.1500
- **F1-Score**: 0.8144

## Usage

### 1. Training
```powershell
python src/scripts/train.py --data_dir <path_to_celebdf> --model_type xception --epochs 10
```

### 2. Full Evaluation
Compute AUC, F1, and EER on the test split:
```powershell
python src/scripts/run_full_eval.py --model_path checkpoints/xception_best.pth --data_dir Celeb-DF-v2
```

### 3. Grad-CAM Visualization
Generate saliency maps to visualize model focus:
```powershell
python src/scripts/run_gradcam_vis.py --video_path <video_path> --model_path checkpoints/xception_best.pth
```

### 4. Robustness Testing
Evaluate model stability against compression, noise, and resizing:
```powershell
python src/scripts/run_robustness_test.py --model_path checkpoints/xception_best.pth --num_samples 20
```

## Robustness Insights
- **Stable**: JPEG Compression (Q=50), Resizing (0.5x).
- **Sensitive**: Gaussian Noise significantly degrades performance (AUC drops to ~0.54).

## Cleanup & Ignored Files
The `.gitignore` is configured to exclude:
- `Celeb-DF-v2/` (Dataset)
- `results/` (Inference artifacts & checkpoints)
- `logs/` (Training logs)
- `__pycache__` and other temp artifacts.
