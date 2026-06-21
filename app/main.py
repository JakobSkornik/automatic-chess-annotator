import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.cleanup import cleanup_old_files
from app.core.worker import analysis_worker
from app.routes import jobs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start worker and cleanup tasks
    worker_task = asyncio.create_task(analysis_worker())
    cleanup_task = asyncio.create_task(cleanup_old_files("data/games"))

    yield

    # Shutdown
    worker_task.cancel()
    cleanup_task.cancel()
    try:
        await worker_task
        await cleanup_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Annotator API", version="0.1.0", lifespan=lifespan)

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)


@app.get("/")
def read_root():
    """Liveness root endpoint."""
    return {"message": "API is running."}


@app.get("/health")
def health():
    """Health-check endpoint."""
    return {"ok": True}
