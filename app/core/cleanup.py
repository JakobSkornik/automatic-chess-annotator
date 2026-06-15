import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

# Default retention for analyzed games: 3 months.
DEFAULT_MAX_AGE_SECONDS = 90 * 24 * 60 * 60


async def cleanup_old_files(
    directory: str, max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS
):
    """
    Deletes files in directory older than max_age_seconds.
    Runs indefinitely with a sleep interval.
    """
    logger.info(f"Starting cleanup task for {directory}, max age {max_age_seconds}s")
    os.makedirs(directory, exist_ok=True)

    while True:
        try:
            now = time.time()
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    file_age = now - os.path.getmtime(filepath)
                    if file_age > max_age_seconds:
                        try:
                            os.remove(filepath)
                            logger.info(f"Deleted old file: {filename}")
                        except Exception as e:
                            logger.error(f"Failed to delete {filename}: {e}")

            await asyncio.sleep(600)  # Check every 10 minutes
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(600)
