import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any

from starlette.websockets import WebSocket

from app.core.io.pgn_reader import PGNReader
from app.models.job import JobResponse, JobStatus
from app.models.PgnMetadata import PgnMetadata

logger = logging.getLogger(__name__)


def _extract_pgn_snapshot(pgn_string: str) -> tuple[PgnMetadata | None, int]:
    """Parse PGN once at enqueue time for job sidebar / recent list."""
    try:
        game = PGNReader.read_game_from_string(pgn_string)
    except ValueError:
        return None, 0
    if game is None:
        return None, 0
    h = game.headers
    we = h.get("WhiteElo", "")
    be = h.get("BlackElo", "")
    white_elo: int | None = None
    black_elo: int | None = None
    try:
        if we and str(we).isdigit():
            white_elo = int(we)
    except (TypeError, ValueError):
        pass
    try:
        if be and str(be).isdigit():
            black_elo = int(be)
    except (TypeError, ValueError):
        pass
    opening = h.get("Opening", "") or ""
    eco = h.get("ECO", "")
    if eco and opening:
        opening = f"{opening} ({eco})" if eco not in opening else opening
    elif eco and not opening:
        opening = eco
    meta = PgnMetadata(
        whiteName=h.get("White", ""),
        blackName=h.get("Black", ""),
        whiteElo=white_elo,
        blackElo=black_elo,
        event=h.get("Event", ""),
        opening=opening,
        eco=eco or None,
        result=h.get("Result", "*"),
    )
    n = 0
    node = game
    while node.variations:
        node = node.variations[0]
        n += 1
    return meta, n


class QueueManager:
    def __init__(self):
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.jobs: dict[str, dict] = {}  # job_id -> job dict
        self.ws_connections: dict[str, list[WebSocket]] = {}
        self.commentary_buffer: dict[str, list[dict[str, Any]]] = {}
        self._enqueue_seq: int = 0

    def _compute_queued_ahead(self, job_id: str) -> int | None:
        job = self.jobs.get(job_id)
        if not job or job["status"] != JobStatus.WAITING:
            return None
        waiting = [j for j in self.jobs.values() if j["status"] == JobStatus.WAITING]
        waiting_sorted = sorted(
            waiting, key=lambda j: (j["created_at"], j.get("enqueue_seq", 0))
        )
        for i, j in enumerate(waiting_sorted):
            if j["id"] == job_id:
                return i
        return None

    def _job_to_response(self, job: dict) -> JobResponse:
        q_ahead = self._compute_queued_ahead(job["id"])
        qp = (q_ahead + 1) if q_ahead is not None else job.get("queue_position")
        return JobResponse(
            job_id=job["id"],
            status=job["status"],
            queue_position=qp,
            progress=job.get("progress"),
            message=job.get("message"),
            created_at=job["created_at"],
            pgn_headers=job.get("pgn_headers"),
            move_count=job.get("move_count"),
            llm_provider=job.get("llm_provider"),
            llm_effort=job.get("llm_effort"),
            commentary_level=job.get("commentary_level"),
            comment_side=job.get("comment_side"),
            error=job.get("error"),
            queued_ahead=q_ahead,
            pgn_preview=job.get("pgn_preview"),
        )

    async def add_job(
        self,
        pgn_string: str,
        llm_provider: str | None = None,
        llm_effort: str | None = None,
        commentary_level: str | None = None,
        comment_side: str | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        self._enqueue_seq += 1
        seq = self._enqueue_seq
        pgn_headers, move_count = _extract_pgn_snapshot(pgn_string)
        job_data = {
            "id": job_id,
            "enqueue_seq": seq,
            "pgn": pgn_string,
            "status": JobStatus.WAITING,
            "created_at": time.time(),
            "progress": 0.0,
            "queue_position": self.job_queue.qsize(),
            "message": "Waiting in queue",
            "llm_provider": llm_provider,
            "llm_effort": llm_effort,
            "commentary_level": commentary_level,
            "comment_side": comment_side,
            "pgn_headers": pgn_headers,
            "move_count": move_count,
            "error": None,
            "pgn_preview": (pgn_string[:500] + "…")
            if len(pgn_string) > 500
            else pgn_string,
        }
        self.jobs[job_id] = job_data
        await self.job_queue.put(job_id)
        logger.info(f"Job {job_id} added to queue.")
        return job_id

    def get_job_status(self, job_id: str) -> JobResponse | None:
        job = self.jobs.get(job_id)
        if not job:
            return None
        return self._job_to_response(job)

    def list_jobs(self, limit: int = 20) -> list[JobResponse]:
        items = sorted(
            self.jobs.values(),
            key=lambda j: (-j["created_at"], -j.get("enqueue_seq", 0)),
        )
        return [self._job_to_response(j) for j in items[:limit]]

    def update_job_status(
        self, job_id: str, status: JobStatus, progress: float = 0.0, message: str = ""
    ):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["progress"] = progress
            self.jobs[job_id]["message"] = message

    def get_job_data(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def mark_failed(self, job_id: str, error: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = JobStatus.FAILED
            self.jobs[job_id]["progress"] = 0.0
            self.jobs[job_id]["message"] = error
            self.jobs[job_id]["error"] = error

    def buffer_commentary(
        self, job_id: str, msg_type: str, payload: dict[str, Any]
    ) -> None:
        """Store commentary messages for late-connecting WebSocket clients."""
        entry = {"type": msg_type, "payload": payload}
        self.commentary_buffer.setdefault(job_id, []).append(entry)

    def get_buffered_commentary(self, job_id: str) -> list[dict[str, Any]]:
        return list(self.commentary_buffer.get(job_id, []))

    def clear_commentary_buffer(self, job_id: str) -> None:
        self.commentary_buffer.pop(job_id, None)

    def add_job_ws(self, job_id: str, ws: WebSocket) -> None:
        self.ws_connections.setdefault(job_id, []).append(ws)

    def remove_job_ws(self, job_id: str, ws: WebSocket) -> None:
        conns = self.ws_connections.get(job_id)
        if not conns:
            return
        with contextlib.suppress(ValueError):
            conns.remove(ws)
        if not conns:
            self.ws_connections.pop(job_id, None)

    async def broadcast_to_job_ws(
        self, job_id: str, msg_type: str, payload: dict[str, Any]
    ) -> None:
        message = {"type": msg_type, "payload": payload}
        for ws in list(self.ws_connections.get(job_id, [])):
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.debug("Removing dead WS for job %s: %s", job_id, e)
                self.remove_job_ws(job_id, ws)


queue_manager = QueueManager()
