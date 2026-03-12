# Orivis: Deepfake Detection System

Orivis is a robust deepfake detection system designed to identify manipulated videos and audio using state-of-the-art multimodal fusion and visual saliency mapping (Grad-CAM).

## Features
- **Multimodal Fusion**: Combines video and audio forgery signals for high-accuracy detection.
- **Asynchronous API**: Scalable backend using FastAPI and background tasks for non-blocking inference.
- **Visual Saliency**: Generates Grad-CAM heatmaps to highlight forged regions in suspicious videos.
- **Dockerized**: Easy deployment with single-command setup.

## Project Structure
- `src/api`: FastAPI backend and background job management.
- `src/fusion`: Multimodal weighting and pipeline logic.
- `src/inference`: Service layers for video and audio models.
- `src/video`: Xception-based video forgery detection.
- `src/audio`: Audio spoofing detection services.
- `results/`: Persistent storage for generated heatmaps.

## Getting Started

### Using Docker (Recommended)
Launch the entire system with one command:
```bash
docker-compose up --build
```
The API will be available at `http://localhost:8000`.

### Local Development
1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```
2. **Run API**:
```bash
python src/api/main.py
```

## API Usage

### 1. Submit Detection Job
```bash
curl -X POST "http://localhost:8000/detect" -F "video=@your_video.mp4"
```
Returns a `job_id`.

### 2. Check Status
```bash
curl "http://localhost:8000/job/{job_id}"
```

### 3. Retrieve Heatmap
If a video is detected as suspicious, a heatmap is generated:
`http://localhost:8000/results/heatmap_{job_id}.jpg`

## CLI Tools
Compute AUC, F1, and EER on the original Celeb-DF v2 test split:
```bash
python src/scripts/run_full_eval.py --model_path models/video_baseline.pth --data_dir Celeb-DF-v2
```




