from pydantic import BaseModel
from typing import Optional
from enum import Enum

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



