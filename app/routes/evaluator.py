import asyncio
import os
import logging
import platform
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict

from app.core.engine.engine_connector import EngineConnector, global_engine_connector
from app.core.io.pgn_reader import PGNReader
from app.core.sessions.analysis_session import AnalysisSession
from app.core.websocket.ws_manager import send_ws_error

from app.models.http.submit_pgn_request import SubmitPgnRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluator", tags=["Evaluator"])

active_sessions: Dict[str, AnalysisSession] = {}


@router.post("/submit_pgn")
async def submit_pgn_for_analysis(request_data: SubmitPgnRequest):
    pgn_string = request_data.pgn_string
    if not pgn_string:
        raise HTTPException(status_code=400, detail="PGN string is required.")

    try:
        PGNReader.validate_single_game(pgn_string)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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


@router.get("/status")
async def get_backend_status():
    try:
        engine_info = {
            "engine_path": global_engine_connector.engine_path,
            "hash": global_engine_connector.hash_size,
            "threads": global_engine_connector.threads,
        }
    except Exception as e:
        engine_info = {"error": str(e)}

    try:
        sessions_count = len(active_sessions)
    except Exception:
        sessions_count = 0

    try:
        ok = True if global_engine_connector.engine is not None else False
    except Exception:
        ok = False

    return {"ok": ok, "engine": engine_info, "active_sessions": sessions_count}


@router.websocket("/ws/analysis/{session_id}")
async def websocket_analysis_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    logger.info(f"WebSocket connection accepted for session_id: {session_id}")

    analysis_session = active_sessions.get(session_id)
    if not analysis_session:
        await send_ws_error(ws, "Analysis session not found.", close_connection=True)
        return

    await analysis_session.add_websocket(ws)

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
        if analysis_session:
            await analysis_session.remove_websocket(ws)
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
