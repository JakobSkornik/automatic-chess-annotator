import asyncio
import logging
import traceback
import json
import os
from app.core.queue_manager import queue_manager
from app.models.job import JobStatus
from app.core.engine.analysis_retriever import AnalysisRetriever
from app.core.engine.engine_connector import global_engine_connector
from app.core.io.pgn_reader import PGNReader
from app.models.GameJson import GameJson

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
                
                # Perform Analysis
                # We will implement run_full_analysis_to_json next
                from app.core.engine.analysis_retriever import run_full_analysis_to_json
                
                # Create a progress callback
                async def progress_callback(percentage: float, message: str):
                    queue_manager.update_job_status(job_id, JobStatus.PROCESSING, progress=percentage, message=message)
                    # Allow context switch to handle cancellation
                    await asyncio.sleep(0)

                game_json: GameJson = await run_full_analysis_to_json(
                    pgn_string, 
                    global_engine_connector, 
                    progress_callback
                )
                
                # Save to file
                output_dir = "data/games"
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"{job_id}.json")
                
                with open(file_path, "w") as f:
                    f.write(game_json.model_dump_json(indent=2))

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
