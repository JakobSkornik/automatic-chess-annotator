import asyncio
import logging
import traceback
import os
from app.core.queue_manager import queue_manager
from app.models.job import JobStatus
from app.core.engine.engine_connector import global_engine_connector
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.tantivy_positional_retriever import get_default_retriever
from app.core.engine.analysis_retriever import (
    assemble_game_json,
    run_engine_analysis_to_json,
    run_llm_commentary,
)

logger = logging.getLogger(__name__)

async def analysis_worker():
    logger.info("Worker started, waiting for jobs...")
    while True:
        try:
            # Check for cancellation before waiting for a job
            try:
                # Use a timeout so we can periodically check for cancellation
                # if the queue is empty.
                job_id = await asyncio.wait_for(queue_manager.job_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("Worker cancelled while waiting for job.")
                break

            job_data = queue_manager.get_job_data(job_id)
            
            if not job_data:
                queue_manager.job_queue.task_done()
                continue
            
            logger.info(f"Processing job {job_id}")
            queue_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=0, message="Starting analysis...")

            try:
                pgn_string = job_data["pgn"]
                
                async def progress_callback(percentage: float, message: str):
                    queue_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=percentage, message=message)
                    await asyncio.sleep(0)

                _, state = await run_engine_analysis_to_json(
                    pgn_string,
                    global_engine_connector,
                    progress_callback,
                    metadata_id=job_id,
                )

                output_dir = "data/games"
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"{job_id}.json")

                game_json_phase1 = assemble_game_json(state)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(game_json_phase1.model_dump_json(indent=2))

                queue_manager.update_job_status(
                    job_id,
                    JobStatus.ENGINE_COMPLETE,
                    progress=95,
                    message="Engine analysis done, generating commentary...",
                )

                advanced_commenter = AdvancedCommentService(rag_retriever=get_default_retriever())

                async def commentary_callback(msg_type: str, payload: dict):
                    queue_manager.buffer_commentary(job_id, msg_type, payload)
                    await queue_manager.broadcast_to_job_ws(job_id, msg_type, payload)

                async def llm_progress(percentage: float, message: str):
                    queue_manager.update_job_status(
                        job_id,
                        JobStatus.ENGINE_COMPLETE,
                        progress=percentage,
                        message=message,
                    )
                    await asyncio.sleep(0)

                await run_llm_commentary(
                    state,
                    advanced_commenter,
                    progress_callback=llm_progress,
                    commentary_callback=commentary_callback,
                    llm_model=job_data.get("llm_model") or os.environ.get("LLM_DEFAULT_MODEL"),
                    llm_effort=job_data.get("llm_effort") or os.environ.get("LLM_DEFAULT_EFFORT"),
                )

                final_json = assemble_game_json(state)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(final_json.model_dump_json(indent=2))

                queue_manager.clear_commentary_buffer(job_id)
                queue_manager.update_job_status(job_id, JobStatus.COMPLETED, progress=100, message="Analysis complete")
                
            except asyncio.CancelledError:
                logger.info(f"Job {job_id} cancelled.")
                queue_manager.mark_failed(job_id, "Analysis cancelled by server shutdown.")
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
