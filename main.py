import os
import shutil
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv
load_dotenv() # Load environment variables from a .env file

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agents import VibeMLOrchestrator

# Directories Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

app = FastAPI(title="VibeML - AutoML Agentic Dashboard")

# Global state to store the pipeline's status, logs, and results
pipeline_status = {
    "status": "idle",  # idle, running, completed, failed
    "logs": [],
    "results": None
}

def log_pipeline_step(message: str):
    """Callback function to record execution logs."""
    print(message)
    pipeline_status["logs"].append(message)

async def run_automl_background(filepath: str, target: str):
    pipeline_status["status"] = "running"
    pipeline_status["logs"] = []
    pipeline_status["results"] = None
    
    try:
        log_pipeline_step(f"Initializing VibeML Orchestrator for dataset: {os.path.basename(filepath)}")
        orchestrator = VibeMLOrchestrator(filepath, OUTPUT_DIR)
        
        # Run pipeline
        results = await orchestrator.run_pipeline(
            target_column=target,
            log_callback=log_pipeline_step
        )
        
        pipeline_status["status"] = "completed"
        pipeline_status["results"] = results
        log_pipeline_step("Success! Model trained and evaluation report compiled.")
    except Exception as e:
        pipeline_status["status"] = "failed"
        log_pipeline_step(f"CRITICAL ERROR: {str(e)}")


class AnalyzeRequest(BaseModel):
    filename: str
    target: str

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a CSV dataset."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "message": "Dataset uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/analyze")
async def start_analysis(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Starts the AutoML pipeline in a background task."""
    filepath = os.path.join(UPLOAD_DIR, request.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
        
    if pipeline_status["status"] == "running":
        raise HTTPException(status_code=400, detail="An analysis is already running.")
        
    # Queue background task
    background_tasks.add_task(run_automl_background, filepath, request.target)
    return {"message": "AutoML pipeline started in the background."}


@app.get("/api/status")
async def get_status():
    """Gets current status and execution logs."""
    return pipeline_status


@app.get("/api/results")
async def get_results():
    """Returns final reports and assets list."""
    if pipeline_status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Results are not ready.")
        
    # Find generated plots
    plots = []
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".png"):
            plots.append(filename)
            
    has_model = os.path.exists(os.path.join(OUTPUT_DIR, "best_model.joblib"))
    
    return {
        "report": pipeline_status["results"]["report"],
        "plots": plots,
        "has_model": has_model
    }


@app.get("/api/plots/{filename}")
async def get_plot(filename: str):
    """Serves generated plots."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(filepath)


@app.get("/api/download-model")
async def download_model():
    """Serves the trained model joblib file."""
    filepath = os.path.join(OUTPUT_DIR, "best_model.joblib")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(filepath, filename="best_model.joblib", media_type="application/octet-stream")


# Mount frontend static files
# Make sure index.html is served from root
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
