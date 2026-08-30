import json
import os

from fastapi import APIRouter, HTTPException, Query, WebSocket, status
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from app.core.io.pgn_reader import PGNReader
from app.core.queue_manager import queue_manager
from app.models.GameJson import GameJson
from app.models.job import JobResponse, JobStatus

router = APIRouter(prefix="/jobs", tags=["Jobs"])


class SubmitPgnRequest(BaseModel):
    pgn_string: str
    llm_provider: str | None = None
    llm_effort: str | None = None
    # Audience register the comments are generated at (pre-analysis choice).
    commentary_level: str | None = None  # beginner | intermediate | expert
    # When set, only this side's moves receive commentary.
    comment_side: str | None = None  # white | black | both


@router.get("", response_model=list[JobResponse])
async def list_jobs(limit: int = Query(20, ge=1, le=100)):
    return queue_manager.list_jobs(limit)


@router.post("/submit", response_model=JobResponse)
async def submit_job(request: SubmitPgnRequest):
    if not request.pgn_string:
        raise HTTPException(status_code=400, detail="PGN string is required")

    try:
        PGNReader.validate_single_game(request.pgn_string)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    level = (request.commentary_level or "").strip().lower() or None
    if level is not None and level not in ("beginner", "intermediate", "expert"):
        raise HTTPException(
            status_code=400,
            detail="commentary_level must be beginner|intermediate|expert",
        )
    side = (request.comment_side or "").strip().lower() or None
    if side is not None and side not in ("white", "black", "both"):
        raise HTTPException(
            status_code=400, detail="comment_side must be white|black|both"
        )

    job_id = await queue_manager.add_job(
        request.pgn_string,
        llm_provider=request.llm_provider,
        llm_effort=request.llm_effort,
        commentary_level=level,
        comment_side=side,
    )
    st = queue_manager.get_job_status(job_id)
    if not st:
        raise HTTPException(status_code=500, detail="Failed to create job")
    return st


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job(job_id: str):
    job = queue_manager.get_job_data(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JobStatus.FAILED:
        raise HTTPException(status_code=400, detail="Can only retry failed jobs")
    new_id = await queue_manager.add_job(
        job["pgn"],
        llm_provider=job.get("llm_provider"),
        llm_effort=job.get("llm_effort"),
        commentary_level=job.get("commentary_level"),
        comment_side=job.get("comment_side"),
    )
    st = queue_manager.get_job_status(new_id)
    if not st:
        raise HTTPException(status_code=500, detail="Failed to create retry job")
    return st


@router.get("/{job_id}/status", response_model=JobResponse)
async def get_job_status(job_id: str):
    st = queue_manager.get_job_status(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    return st


@router.get("/{job_id}/game", response_model=GameJson)
async def get_job_game(job_id: str):
    file_path = f"data/games/{job_id}.json"
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error reading game file: {e}"
            ) from e

    job_status = queue_manager.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    if job_status.status == JobStatus.FAILED:
        raise HTTPException(status_code=400, detail=f"Job failed: {job_status.message}")
    raise HTTPException(status_code=400, detail="Job not ready yet")


@router.get("/export.zip")
async def export_all_games():
    """Zip of every finished game JSON (review repository for annotators)."""
    import io
    import zipfile

    from fastapi.responses import Response

    games_dir = "data/games"
    files = (
        sorted(f for f in os.listdir(games_dir) if f.endswith(".json"))
        if os.path.isdir(games_dir)
        else []
    )
    if not files:
        raise HTTPException(status_code=404, detail="No finished games to export")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in files:
            zf.write(os.path.join(games_dir, fn), fn)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="annotated-games.zip"'},
    )


@router.get("/{job_id}/pgn")
async def get_job_pgn(job_id: str):
    """Annotated PGN export: comments, [%eval] tags, NAGs and variations."""
    from fastapi.responses import PlainTextResponse

    from app.core.io.pgn_writer import game_json_to_pgn

    file_path = f"data/games/{job_id}.json"
    if not os.path.exists(file_path):
        job_status = queue_manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        if job_status.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=400, detail=f"Job failed: {job_status.message}"
            )
        raise HTTPException(status_code=400, detail="Job not ready yet")
    try:
        with open(file_path, encoding="utf-8") as f:
            gj = GameJson.model_validate(json.load(f))
        pgn = game_json_to_pgn(gj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PGN export failed: {e}") from e
    return PlainTextResponse(
        pgn,
        media_type="application/x-chess-pgn",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.pgn"'},
    )


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
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except RuntimeError:
        pass
    finally:
        queue_manager.remove_job_ws(job_id, websocket)
