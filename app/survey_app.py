"""Read saved games and collect survey responses without starting analysis workers.

Deploy with: uvicorn app.survey_app:app --host 0.0.0.0 --port 8000
"""
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from app.routes.surveys import router
from app.routes.survey_grid import router as grid_router
from app.survey_bundle import install_bundled_surveys
from app.drive_backup import backup_quietly


@asynccontextmanager
async def lifespan(app: FastAPI):
    install_bundled_surveys()
    # Catch up responses whose Drive upload failed before the last restart.
    asyncio.get_running_loop().run_in_executor(None, backup_quietly)
    yield


app = FastAPI(title='Chess commentary survey', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.include_router(router)
app.include_router(grid_router)


@app.get('/health')
def health():
    return {'ok': True}
