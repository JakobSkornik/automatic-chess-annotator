from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import json
from app.core.queue_manager import queue_manager
from app.models.job import JobResponse
from app.models.GameJson import GameJson

router = APIRouter(prefix="/jobs", tags=["Jobs"])

class SubmitPgnRequest(BaseModel):
    pgn_string: str

@router.post("/submit", response_model=JobResponse)
async def submit_job(request: SubmitPgnRequest):
    if not request.pgn_string:
        raise HTTPException(status_code=400, detail="PGN string is required")
    
    job_id = await queue_manager.add_job(request.pgn_string)
    status = queue_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=500, detail="Failed to create job")
    return status

@router.get("/{job_id}/status", response_model=JobResponse)
async def get_job_status(job_id: str):
    status = queue_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

@router.get("/{job_id}/game", response_model=GameJson)
async def get_job_game(job_id: str):
    # Check if file exists
    file_path = f"data/games/{job_id}.json"
    if not os.path.exists(file_path):
        # Check if job exists and failed
        status = queue_manager.get_job_status(job_id)
        if status and status.status == "failed":
             raise HTTPException(status_code=400, detail=f"Job failed: {status.message}")
        if status and status.status != "completed":
             raise HTTPException(status_code=400, detail="Job not completed yet")
        raise HTTPException(status_code=404, detail="Game file not found")
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading game file: {e}")



