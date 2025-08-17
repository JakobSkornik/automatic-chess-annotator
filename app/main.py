import logging
from fastapi import FastAPI
from app.routes import evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Annotator API", version="0.1.0")

app.include_router(evaluator.router)


@app.get("/")
def read_root():
    return {"message": "API is running."}
