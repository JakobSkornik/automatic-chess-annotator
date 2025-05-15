import os
from fastapi import APIRouter, HTTPException

from app.core.io.pgn_reader import PGNReader
from app.core.engine.engine_connector import EngineConnector
from app.core.engine.analysis_retriever import AnalysisRetriever
from app.models.AnalysisResponse import AnalysisResponse
from app.models.EvaluationRequest import EvaluationRequest

router = APIRouter(prefix="/evaluator", tags=["Evaluator"])

@router.post("/")
def evaluate(request: EvaluationRequest) -> AnalysisResponse:
    # try:
    # Load the PGN game
    game = PGNReader.read_game_from_string(request.pgn)
    if not game:
        raise ValueError("Failed to parse PGN data.")

    # Initialize engine connector and analysis retriever
    stockfish_path = os.path.join(os.path.dirname(__file__), "..", "stockfish")
    with EngineConnector(stockfish_path, default_time_limit=0.1) as connector:
        analysis_retriever = AnalysisRetriever(connector, shallow_depth=4, deep_depth=12)
        analysis = analysis_retriever.retrieve_analysis(game)

    return analysis
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail=str(e))
