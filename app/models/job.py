from pydantic import BaseModel
from typing import Optional
from enum import Enum

from app.models.PgnMetadata import PgnMetadata


class JobStatus(str, Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    ENGINE_COMPLETE = "engine_complete"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    queue_position: Optional[int] = None
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: float
    pgn_headers: Optional[PgnMetadata] = None
    move_count: Optional[int] = None
    llm_model: Optional[str] = None
    llm_effort: Optional[str] = None
    error: Optional[str] = None
    queued_ahead: Optional[int] = None
    pgn_preview: Optional[str] = None
