import asyncio
import time
import uuid
import logging
from typing import Dict, Optional
from app.models.job import JobStatus, JobResponse

logger = logging.getLogger(__name__)

class QueueManager:
    def __init__(self):
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.jobs: Dict[str, Dict] = {} # job_id -> {status, progress, created_at, ...}

    async def add_job(self, pgn_string: str) -> str:
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "pgn": pgn_string,
            "status": JobStatus.WAITING,
            "created_at": time.time(),
            "progress": 0.0,
            "queue_position": self.job_queue.qsize(),
            "message": "Waiting in queue"
        }
        self.jobs[job_id] = job_data
        await self.job_queue.put(job_id)
        logger.info(f"Job {job_id} added to queue.")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[JobResponse]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        # Calculate dynamic queue position if waiting
        queue_pos = None
        if job["status"] == JobStatus.WAITING:
             # This is a simplification; for a real queue position we might need to iterate or track index
             # For now, just None or estimated
             pass 

        return JobResponse(
            job_id=job["id"],
            status=job["status"],
            queue_position=queue_pos, # Simplified
            progress=job.get("progress"),
            message=job.get("message"),
            created_at=job["created_at"]
        )
    
    def update_job_status(self, job_id: str, status: JobStatus, progress: float = 0.0, message: str = ""):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["progress"] = progress
            self.jobs[job_id]["message"] = message

    def get_job_data(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)

    def mark_failed(self, job_id: str, error: str):
        self.update_job_status(job_id, JobStatus.FAILED, message=error)

queue_manager = QueueManager()



