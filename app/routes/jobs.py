from fastapi import APIRouter, HTTPException, WebSocket, status
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import os
import json
from app.core.queue_manager import queue_manager
from app.core.io.pgn_reader import PGNReader
from app.models.job import JobResponse, JobStatus
from app.models.GameJson import GameJson

router = APIRouter(prefix="/jobs", tags=["Jobs"])

class SubmitPgnRequest(BaseModel):
    pgn_string: str
    llm_model: Optional[str] = None
    llm_effort: Optional[str] = None

@router.post("/submit", response_model=JobResponse)
async def submit_job(request: SubmitPgnRequest):
    if not request.pgn_string:
        raise HTTPException(status_code=400, detail="PGN string is required")

    try:
        PGNReader.validate_single_game(request.pgn_string)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = await queue_manager.add_job(
        request.pgn_string,
        llm_model=request.llm_model,
        llm_effort=request.llm_effort,
    )
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
    file_path = f"data/games/{job_id}.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading game file: {e}")

    job_status = queue_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_status.status == JobStatus.FAILED:
        raise HTTPException(status_code=400, detail=f"Job failed: {job_status.message}")
    raise HTTPException(status_code=400, detail="Job not ready yet")


@router.websocket("/{job_id}/ws")
async def job_commentary_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    job = queue_manager.get_job_data(job_id)
    if not job:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    st = job.get("status")
    if st == JobStatus.FAILED:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    queue_manager.add_job_ws(job_id, websocket)
    try:
        for msg in queue_manager.get_buffered_commentary(job_id):
            await websocket.send_json(msg)
        while True:
            await websocket.receive()
    except WebSocketDisconnect:
        pass
    finally:
        queue_manager.remove_job_ws(job_id, websocket)



