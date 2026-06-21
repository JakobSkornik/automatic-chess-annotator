import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

# Default retention for analyzed games: 3 months.
DEFAULT_MAX_AGE_SECONDS = 90 * 24 * 60 * 60
CLEANUP_INTERVAL_SECONDS = 600  # re-scan every 10 minutes


def _remove_stale_files(directory: str, now: float, max_age_seconds: int) -> None:
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
        if now - os.path.getmtime(filepath) <= max_age_seconds:
            continue
        try:
            os.remove(filepath)
            logger.info(f"Deleted old file: {filename}")
        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")


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
            _remove_stale_files(directory, time.time(), max_age_seconds)
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
