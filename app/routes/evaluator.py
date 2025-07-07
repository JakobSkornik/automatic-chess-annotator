import asyncio
import os
import logging
import platform
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict

from app.core.engine.engine_connector import EngineConnector
from app.core.sessions.analysis_session import AnalysisSession
from app.core.websocket.ws_manager import send_ws_error

from app.models.http.submit_pgn_request import SubmitPgnRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluator", tags=["Evaluator"])

active_sessions: Dict[str, AnalysisSession] = {}

# Determine Stockfish path based on OS
stockfish_executable = (
    "stockfish.exe" if platform.system() == "Windows" else "stockfish"
)
stockfish_path = os.path.join(os.path.dirname(__file__), "..", stockfish_executable)
global_engine_connector = EngineConnector(stockfish_path)


@router.post("/submit_pgn")
async def submit_pgn_for_analysis(request_data: SubmitPgnRequest):
    pgn_string = request_data.pgn_string
    if not pgn_string:
        raise HTTPException(status_code=400, detail="PGN string is required.")

    session_id = str(uuid.uuid4())
    try:
        analysis_session = AnalysisSession(
            engine_connector=global_engine_connector,
            pgn_game_str=pgn_string,
            session_id=session_id,
        )
        active_sessions[session_id] = analysis_session
    except Exception as e:
        detail = f"Failed to create AnalysisSession {session_id}: {e}"
        logger.error(detail, exc_info=True)
        raise HTTPException(status_code=500, detail=detail)

    logger.info(f"Analysis session created for session_id: {session_id}")
    return {
        "session_id": session_id,
        "websocket_url": f"/ws/analysis/{session_id}",
    }


@router.websocket("/ws/analysis/{session_id}")
async def websocket_analysis_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    logger.info(f"WebSocket connection accepted for session_id: {session_id}")

    analysis_session = active_sessions.get(session_id)
    if not analysis_session:
        await send_ws_error(ws, "Analysis session not found.", close_connection=True)
        return

    try:
        ws_input_task = asyncio.create_task(analysis_session.process_ws_message(ws))
        process_task = asyncio.create_task(analysis_session.process_requests(ws))
        await asyncio.gather(ws_input_task, process_task)
    except Exception as e:
        detail = f"Unhandled exception in websocket_analysis_endpoint for session {session_id}: {e}"
        logger.error(detail, exc_info=True)

        if not ws.client_state == WebSocketDisconnect:
            try:
                await send_ws_error(
                    ws, "A critical server error occurred.", close_connection=True
                )
            except Exception:
                detail = f"Failed to send final unhandled error message to WebSocket for session {session_id}."
                logger.error(detail)
    finally:
        if session_id in active_sessions:
            current_session_to_close = active_sessions.pop(session_id)
            try:
                await current_session_to_close.close_session()
                logger.info(f"AnalysisSession {session_id} closed and cleaned up.")
            except Exception as e_close:
                detail = f"Error closing session {session_id}: {e_close}"
                logger.error(detail, exc_info=True)
        else:
            detail = f"Session {session_id} not found in active sessions for cleanup."
            logger.info(detail)
        logger.info(f"WebSocket session {session_id} processing fully ended.")
