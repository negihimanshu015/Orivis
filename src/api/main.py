from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import cv2
import shutil
from typing import Dict, Any

from src.fusion.fusion_pipeline import FusionPipeline
from src.api.utils import save_temp_file

app = FastAPI(
    title="Orivis Deepfake Detection API",
    description="Multimodal (Video + Audio) Deepfake Detection Service",
    version="1.1.0"
)

# In-memory job store (for demonstration, replace with DB/Redis for production)
jobs: Dict[str, Any] = {}

# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Mount static files for Grad-CAM heatmaps
app.mount("/results", StaticFiles(directory="results"), name="results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    pipeline = FusionPipeline(video_weight=0.6, audio_weight=0.4)
except Exception as e:
    print(f"Error initializing FusionPipeline: {e}")
    pipeline = None

@app.get("/")
async def root():
    return {"message": "Orivis API is running", "status": "healthy", "version": "1.1.0"}

def process_detection(job_id: str, video_tmp_path: str, audio_tmp_path: str = None):
    """
    Background task to run inference.
    """
    try:
        jobs[job_id]["status"] = "processing"
        
        # Run fusion pipeline
        results = pipeline.run_fusion(video_tmp_path, audio_tmp_path)
        
        # Save heatmap if exists
        heatmap = results.get("heatmap")
        if heatmap is not None:
            heatmap_filename = f"heatmap_{job_id}.jpg"
            heatmap_path = os.path.join("results", heatmap_filename)
            # Convert RGB to BGR for OpenCV
            heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            cv2.imwrite(heatmap_path, heatmap_bgr)
            results["heatmap_url"] = f"/results/{heatmap_filename}"
            # Remove raw heatmap array from response
            del results["heatmap"]
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["results"] = results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Cleanup temp files
        if os.path.exists(video_tmp_path):
            os.remove(video_tmp_path)
        if audio_tmp_path and os.path.exists(audio_tmp_path):
            os.remove(audio_tmp_path)

@app.post("/detect")
async def detect(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...), 
    audio: UploadFile = File(None)
):
    """
    Asynchronously detect synthetic content. Returns a job_id.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Inference pipeline not initialized")

    job_id = str(uuid.uuid4())
    
    # Save files to a semi-persistent temp location for the background task
    video_suffix = os.path.splitext(video.filename)[1]
    video_tmp_path = os.path.join("results", f"tmp_v_{job_id}{video_suffix}")
    with open(video_tmp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
        
    audio_tmp_path = None
    if audio:
        audio_suffix = os.path.splitext(audio.filename)[1]
        audio_tmp_path = os.path.join("results", f"tmp_a_{job_id}{audio_suffix}")
        with open(audio_tmp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

    jobs[job_id] = {"status": "queued", "id": job_id}
    
    background_tasks.add_task(process_detection, job_id, video_tmp_path, audio_tmp_path)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status and results of a detection job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
