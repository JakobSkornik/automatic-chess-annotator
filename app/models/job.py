from enum import Enum

from pydantic import BaseModel

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
    queue_position: int | None = None
    progress: float | None = None
    message: str | None = None
    created_at: float
    pgn_headers: PgnMetadata | None = None
    move_count: int | None = None
    llm_provider: str | None = None
    llm_effort: str | None = None
    commentary_level: str | None = None
    comment_side: str | None = None
    error: str | None = None
    queued_ahead: int | None = None
    pgn_preview: str | None = None
