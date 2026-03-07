from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from src.fusion.fusion_pipeline import FusionPipeline
from src.api.utils import save_temp_file

app = FastAPI(
    title="Orivis Deepfake Detection API",
    description="Multimodal (Video + Audio) Deepfake Detection Service",
    version="1.0.0"
)

             
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

                                        
                                                          
try:
    pipeline = FusionPipeline()
except Exception as e:
    print(f"Error initializing FusionPipeline: {e}")
    pipeline = None

@app.get("/")
async def root():
    return {"message": "Orivis API is running", "status": "healthy"}

@app.post("/detect")
async def detect(
    video: UploadFile = File(...), 
    audio: UploadFile = File(None)
):
    """
    Detect synthetic content in video and optional audio.
    If only video is provided, the system may extract audio from the video.
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Inference pipeline not initialized")

    try:
                                    
        with save_temp_file(video) as video_path:
            
                                                    
            if audio:
                with save_temp_file(audio) as audio_path:
                    results = pipeline.run_fusion(video_path, audio_path)
            else:
                                                                                           
                results = pipeline.run_fusion(video_path)
        
        return results

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
