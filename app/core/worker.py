import asyncio
import logging
import os
import traceback

from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.pipeline.game_pipeline import GameAnnotationPipeline
from app.core.engine.analysis_retriever import (
    assemble_game_json,
    run_engine_analysis_to_json,
)
from app.core.engine.engine_connector import get_global_engine_connector
from app.core.queue_manager import queue_manager
from app.models.job import JobStatus

logger = logging.getLogger(__name__)


GAMES_OUTPUT_DIR = "data/games"
ENGINE_DONE_PROGRESS = 95
JOB_COMPLETE_PROGRESS = 100


def _job_output_path(job_id: str) -> str:
    os.makedirs(GAMES_OUTPUT_DIR, exist_ok=True)
    return os.path.join(GAMES_OUTPUT_DIR, f"{job_id}.json")


def _write_game_json(file_path: str, game_json) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(game_json.model_dump_json(indent=2))


async def _run_engine_phase(job_id: str, pgn_string: str, file_path: str):
    """Engine analysis pass: writes the phase-1 GameJson, returns the state."""

    async def progress_callback(percentage: float, message: str) -> None:
        queue_manager.update_job_status(
            job_id, JobStatus.PROCESSING, progress=percentage, message=message
        )
        await asyncio.sleep(0)

    _, state = await run_engine_analysis_to_json(
        pgn_string,
        get_global_engine_connector(),
        progress_callback,
        metadata_id=job_id,
    )
    _write_game_json(file_path, assemble_game_json(state))
    queue_manager.update_job_status(
        job_id,
        JobStatus.ENGINE_COMPLETE,
        progress=ENGINE_DONE_PROGRESS,
        message="Engine analysis done, generating commentary...",
    )
    return state


async def _run_commentary_phase(
    job_id: str, job_data: dict, state, file_path: str
) -> None:
    """LLM commentary sweep: streams updates, rewrites the final GameJson."""
    prov = (
        (
            job_data.get("llm_provider")
            or os.environ.get("LLM_DEFAULT_PROVIDER")
            or "openai"
        )
        .strip()
        .lower()
    )
    advanced_commenter = AdvancedCommentService(provider_key=prov)

    async def commentary_callback(msg_type: str, payload: dict) -> None:
        queue_manager.buffer_commentary(job_id, msg_type, payload)
        await queue_manager.broadcast_to_job_ws(job_id, msg_type, payload)

    async def llm_progress(percentage: float, message: str) -> None:
        queue_manager.update_job_status(
            job_id, JobStatus.ENGINE_COMPLETE, progress=percentage, message=message
        )
        await asyncio.sleep(0)

    await GameAnnotationPipeline().run_llm_phases(
        state,
        advanced_commenter,
        progress_callback=llm_progress,
        commentary_callback=commentary_callback,
        llm_effort=job_data.get("llm_effort") or os.environ.get("LLM_DEFAULT_EFFORT"),
        commentary_level=job_data.get("commentary_level"),
        comment_side=job_data.get("comment_side"),
    )
    _write_game_json(file_path, assemble_game_json(state))


async def _process_job(job_id: str, job_data: dict) -> None:
    """Full pipeline for one job: engine pass, commentary sweep, completion."""
    file_path = _job_output_path(job_id)
    state = await _run_engine_phase(job_id, job_data["pgn"], file_path)
    await _run_commentary_phase(job_id, job_data, state, file_path)
    queue_manager.clear_commentary_buffer(job_id)
    queue_manager.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        progress=JOB_COMPLETE_PROGRESS,
        message="Analysis complete",
    )


async def _next_job(self_timeout: float = 1.0) -> str | None:
    """Next queued job id, or None on idle timeout (lets the loop re-check)."""
    try:
        return await asyncio.wait_for(
            queue_manager.job_queue.get(), timeout=self_timeout
        )
    except asyncio.TimeoutError:
        return None


async def analysis_worker():
    logger.info("Worker started, waiting for jobs...")
    while True:
        try:
            try:
                job_id = await _next_job()
            except asyncio.CancelledError:
                logger.info("Worker cancelled while waiting for job.")
                break
            if job_id is None:
                continue

            job_data = queue_manager.get_job_data(job_id)
            if not job_data:
                queue_manager.job_queue.task_done()
                continue

            logger.info(f"Processing job {job_id}")
            queue_manager.update_job_status(
                job_id, JobStatus.PROCESSING, progress=0, message="Starting analysis..."
            )
            try:
                await _process_job(job_id, job_data)
            except asyncio.CancelledError:
                logger.info(f"Job {job_id} cancelled.")
                queue_manager.mark_failed(
                    job_id, "Analysis cancelled by server shutdown."
                )
                raise
            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                traceback.print_exc()
                queue_manager.mark_failed(job_id, str(e))
            finally:
                queue_manager.job_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Worker cancelled.")
            break
        except Exception as e:
            logger.error(f"Worker crashed: {e}")
            await asyncio.sleep(1)
