import asyncio
import chess
import logging
import uuid
import time

from fastapi import WebSocket
from typing import List, Optional, Dict

from app.core.engine.analysis_retriever import AnalysisRetriever
from app.core.engine.engine_connector import EngineConnector
from app.core.io.pgn_reader import PGNReader
from app.core.websocket.ws_manager import WebSocketManager, send_ws_message, parse_client_ws_message
from app.core.commentary.commenting_service import CommentingService

from app.models.ws.server_messages import (
    SessionMetadataPayload,
    DetailedAnalysisPayload,
    AnalysisProgressPayload,
    FullAnalysisCompletePayload,
)
from app.models.ws.message_types import ServerMessageType, ClientMessageType
from app.models.ws.server_messages import ModelParamsUpdatedPayload
from app.models.Move import Move

logger = logging.getLogger(__name__)


class SessionJob:
    _sequence_counter = 0

    def __init__(
        self,
        type: ClientMessageType,
        payload: Optional[dict] = {},
        task_prio: int = 3,
        timestamp: Optional[float] = None,
    ):
        self.type = type
        self.payload = payload
        self.task_prio = task_prio
        self.timestamp = timestamp if timestamp is not None else time.time()
        SessionJob._sequence_counter += 1
        self.sequence_num = SessionJob._sequence_counter

    def __lt__(self, other: "SessionJob") -> bool:
        """Compare jobs for priority queue. Lower task_prio is higher priority. If task_prio is equal, earlier timestamp is higher priority."""
        if self.task_prio != other.task_prio:
            return self.task_prio < other.task_prio
        if (
            abs(self.timestamp - other.timestamp) > 0.001
        ):  # Avoid floating point comparison issues
            return self.timestamp < other.timestamp
        return self.sequence_num < other.sequence_num


class AnalysisSession:
    def __init__(
        self,
        pgn_game_str: str,
        engine_connector: EngineConnector,
        session_id: Optional[str] = None,
    ):
        pgn_reader = PGNReader()
        self.session_id: str = session_id or str(uuid.uuid4())
        self.request_queue: asyncio.PriorityQueue[SessionJob] = asyncio.PriorityQueue()

        self.pgn_game_str: str = pgn_game_str
        try:
            self.pgn_game = pgn_reader.read_game_from_string(pgn_game_str)
        except Exception as e:
            detail = f"Session {self.session_id}: Failed to parse PGN string during init: {e}"
            logger.error(detail)
            self.pgn_game = None

        self.engine_connector: EngineConnector = engine_connector
        self.analysis_retriever = AnalysisRetriever(engine_connector, self.pgn_game)
        self.ws_manager = WebSocketManager()
        self.commenting_service = CommentingService(self.ws_manager, self.pgn_game)
        # Default AI model parameters for this session
        self.model_params: Dict = {
            "model": "gpt-5-mini",
            "effort": "low",
            "temperature": 0.2,
            "maxTokens": 120,
        }
        # Initialize commenting service params
        try:
            self.commenting_service.update_model_params(self.model_params)
        except Exception:
            pass
        logger.info(f"AnalysisSession {self.session_id} created.")

        # For trace tree node IDs
        self._node_id_counter: int = 0

        # WebSocket connections
        self.websockets: List[WebSocket] = []
        self._analysis_in_progress = False

        # Persist analyzed results for the session so commentary/key-moments can use full history
        self._move_results_by_depth: Dict[int, Move] = {}
        self._pvs_by_depth: Dict[int, List[List[Move]]] = {}

    async def add_websocket(self, websocket: WebSocket):
        self.websockets.append(websocket)
        self.ws_manager.add_websocket(websocket)

    async def remove_websocket(self, websocket: WebSocket):
        if websocket in self.websockets:
            self.websockets.remove(websocket)
        self.ws_manager.remove_websocket(websocket)

    async def submit_request(self, request: SessionJob):
        """Submits a client request to the processing queue."""
        await self.request_queue.put(request)

    async def process_requests(self, ws: WebSocket):
        """Continuously processes requests from the queue, properly respecting priorities."""
        try:
            while True:
                if self.request_queue.empty():
                    await asyncio.sleep(0.01)
                    continue

                client_message = await self.request_queue.get()

                if not client_message:
                    self.request_queue.task_done()
                    continue

                try:
                    if client_message.type == ClientMessageType.GET_SESSION_METADATA:
                        await self._handle_get_session_metadata(ws)

                    elif client_message.type == ClientMessageType.GET_MOVE_LIST:
                        await self._handle_get_move_list(ws)

                    elif client_message.type == ClientMessageType.GET_DETAILED_ANALYSIS:
                        move = Move(**client_message.payload["move"])
                        stage = client_message.payload["stage"]
                        await self._handle_analyze_move(move, stage, ws)

                    elif client_message.type == ClientMessageType.GET_TRACE_TREE:
                        mainline_moves_data = client_message.payload.get(
                            "mainlineMoves", []
                        )
                        initial_mainline_moves = [
                            Move(**m_data) for m_data in mainline_moves_data
                        ]
                        await self._handle_get_trace_tree(ws, initial_mainline_moves)

                    # Updated to handle the new message type
                    elif client_message.type == ClientMessageType.GET_GAME_ANALYSIS:
                        await self._handle_full_game_analysis(ws)

                    elif client_message.type == ClientMessageType.SET_MODEL_PARAMS:
                        await self._handle_set_model_params(ws, client_message.payload)

                    else:
                        await send_ws_message(
                            ws,
                            ServerMessageType.ERROR,
                            {"message": "Unknown message type."},
                        )
                        self.request_queue.task_done()

                    self.request_queue.task_done()

                except Exception as e:
                    logger.error(f"Error processing message {client_message.type}: {e}")
                    await send_ws_message(
                        ws,
                        ServerMessageType.ERROR,
                        {"message": f"Error processing request: {str(e)}"},
                    )
                    self.request_queue.task_done()

                # Check for new higher priority messages after each processing step
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info(
                f"Session {self.session_id}: Request processing loop cancelled."
            )
        except Exception as e:
            detail = f"Critical error in request processing loop {self.session_id}: {e}"
            logger.error(detail, exc_info=True)
        finally:
            logger.info(f"Session {self.session_id}: Exiting request processing loop.")

    async def process_ws_message(self, ws: WebSocket):
        """Processes incoming WebSocket messages."""
        try:
            while True:
                raw_data = await ws.receive_text()
                ws_msg = await parse_client_ws_message(raw_data, ws)

                if ws_msg:
                    if ws_msg.type == ClientMessageType.GET_SESSION_METADATA:
                        job = SessionJob(
                            type=ws_msg.type,
                            payload=ws_msg.payload,
                            task_prio=1,
                            timestamp=time.time(),
                        )
                        await self.submit_request(job)

                    elif ws_msg.type == ClientMessageType.GET_MOVE_LIST:
                        job = SessionJob(
                            type=ws_msg.type,
                            payload=ws_msg.payload,
                            task_prio=2,
                            timestamp=time.time(),
                        )
                        await self.submit_request(job)

                    elif ws_msg.type == ClientMessageType.GET_DETAILED_ANALYSIS:
                        stages = self._get_analysis_stages()
                        prio = 3
                        for stage in stages:
                            job = SessionJob(
                                type=ClientMessageType.GET_DETAILED_ANALYSIS,
                                payload={
                                    "move": ws_msg.payload["move"],
                                    "stage": stage,
                                },
                                task_prio=prio,
                                timestamp=time.time(),
                            )
                            prio = prio + 1
                            await self.submit_request(job)

                    # Updated to handle the new message type
                    elif ws_msg.type == ClientMessageType.GET_GAME_ANALYSIS:
                        job = SessionJob(
                            type=ClientMessageType.GET_GAME_ANALYSIS,
                            payload=ws_msg.payload,
                            task_prio=1,  # High priority
                            timestamp=time.time(),
                        )
                        # Clear the request queue before starting a full game analysis
                        self.request_queue = asyncio.PriorityQueue()
                        await self.submit_request(job)

                    elif ws_msg.type == ClientMessageType.SET_MODEL_PARAMS:
                        job = SessionJob(
                            type=ClientMessageType.SET_MODEL_PARAMS,
                            payload=ws_msg.payload,
                            task_prio=1,
                            timestamp=time.time(),
                        )
                        await self.submit_request(job)

                    else:
                        await send_ws_message(
                            ws,
                            ServerMessageType.ERROR,
                            {"message": "Unknown message type."},
                        )

        except asyncio.CancelledError:
            logger.info(
                f"Session {self.session_id}: Request processing loop cancelled."
            )
        except Exception as e:
            logger.error(f"Error in WebSocket listening task for session.: {e}")
        finally:
            logger.info(f"Session {self.session_id}: Exiting request processing loop.")

    async def _handle_set_model_params(self, ws: WebSocket, payload: dict):
        try:
            # Update session model params with provided keys
            for key in ["model", "effort", "temperature", "maxTokens"]:
                if key in payload and payload[key] is not None:
                    self.model_params[key] = payload[key]

            # Inform commenting service of updated params
            try:
                self.commenting_service.update_model_params(self.model_params)
            except Exception:
                pass

            # Broadcast updated params to the requesting client
            await send_ws_message(
                ws,
                ServerMessageType.MODEL_PARAMS_UPDATED,
                ModelParamsUpdatedPayload(
                    model=self.model_params.get("model", "gpt-5-mini"),
                    effort=self.model_params.get("effort", "low"),
                    temperature=self.model_params.get("temperature"),
                    maxTokens=self.model_params.get("maxTokens"),
                ).model_dump(),
            )
        except Exception as e:
            await send_ws_message(ws, ServerMessageType.ERROR, {"message": f"Failed to set model params: {e}"})

    async def _handle_get_session_metadata(self, ws: WebSocket):
        """Handles the logic for GET_SESSION_METADATA message type."""
        headers = self.analysis_retriever.get_pgn_headers()
        metadata_payload = SessionMetadataPayload(headers=headers)
        await send_ws_message(ws, ServerMessageType.SESSION_METADATA, metadata_payload)

    async def _handle_get_move_list(self, ws: WebSocket):
        """Handles the logic for GET_MOVE_LIST message type."""
        move_list = self.analysis_retriever.get_move_list()
        move_list_payload = {"moveList": [move.model_dump() for move in move_list]}
        await send_ws_message(ws, ServerMessageType.MOVE_LIST, move_list_payload)

    async def _handle_analyze_move(
        self, move_to_analyze: Move, stage: float, ws: WebSocket
    ):
        """Handles the logic for GET_DETAILED_ANALYSIS message type."""
        analyzed_main_move, pvs_list_of_list_of_moves = (
            self.analysis_retriever.analyze_move(move_to_analyze, stage)
        )

        pvs_for_payload = [
            [pv_move.model_dump() for pv_move in pv_sequence]
            for pv_sequence in pvs_list_of_list_of_moves
        ]

        payload = DetailedAnalysisPayload(
            move=analyzed_main_move.model_dump(), pvs=pvs_for_payload
        )
        await send_ws_message(ws, ServerMessageType.ANALYSIS_UPDATE, payload)

        # Only generate comments for the final analysis stage
        if analyzed_main_move.analysisStage != "final":
            return

        # Emit comment if final stage
        try:
            # Persist analyzed result in the session for future context
            try:
                self._move_results_by_depth[analyzed_main_move.depth] = analyzed_main_move
                self._pvs_by_depth[analyzed_main_move.depth] = pvs_list_of_list_of_moves
            except Exception:
                pass

            # Augment trace with mainline UCI sequence up to this depth for ECO
            try:
                seq = []
                if self.pgn_game is not None:
                    board = self.pgn_game.board()
                    for idx, mv in enumerate(self.pgn_game.mainline_moves()):
                        if idx >= analyzed_main_move.depth:
                            break
                        seq.append(board.uci(mv))
                        board.push(mv)
                if seq:
                    if not analyzed_main_move.trace:
                        analyzed_main_move.trace = {}
                    analyzed_main_move.trace["mainlineUci"] = seq
            except Exception as e:
                logger.error(f"Error augmenting trace: {e}")
            
            # Prefer previously analyzed move stored in this session, fallback to initial move list
            previous_move = self._move_results_by_depth.get(analyzed_main_move.depth - 1) or \
                self.analysis_retriever.get_move_by_depth(analyzed_main_move.depth - 1)

            # Attach score context for AI (prevScore, scoreDelta, scoreTrend) and explicit POV/mover info
            try:
                if not analyzed_main_move.hiddenFeatures:
                    analyzed_main_move.hiddenFeatures = {}
                if isinstance(analyzed_main_move.hiddenFeatures, dict):
                    analyzed_main_move.hiddenFeatures.setdefault("_ai", {})
                    ai_meta = analyzed_main_move.hiddenFeatures["_ai"]
                    if previous_move and previous_move.score is not None and analyzed_main_move.score is not None:
                        ai_meta["prevScore"] = previous_move.score
                        ai_meta["scoreDelta"] = analyzed_main_move.score - previous_move.score
                        ai_meta["scoreNow"] = analyzed_main_move.score
                        ai_meta["scorePov"] = "white"  # engine scores are white POV centipawns
                        try:
                            ai_meta["scoreSwingPawns"] = (ai_meta["scoreDelta"] or 0.0) / 100.0
                            ai_meta["evalBeforePawns"] = (ai_meta["prevScore"] or 0.0) / 100.0
                            ai_meta["evalAfterPawns"] = (ai_meta["scoreNow"] or 0.0) / 100.0
                            ai_meta["swingPawnsAbs"] = abs(ai_meta["scoreSwingPawns"])
                        except Exception:
                            pass
                        # Determine mover and benefited side
                        moved_by_white = (analyzed_main_move.depth % 2 == 1)
                        ai_meta["movedBy"] = "white" if moved_by_white else "black"
                        if isinstance(ai_meta.get("scoreDelta"), (int, float)):
                            benefited = "white" if ai_meta["scoreDelta"] > 0 else ("black" if ai_meta["scoreDelta"] < 0 else None)
                            if benefited is not None:
                                ai_meta["sideBenefited"] = benefited
                                ai_meta["moverMistake"] = (benefited != ai_meta["movedBy"])
                    # Build simple trend from last few stored moves
                    trend: List[float] = []
                    for d in range(analyzed_main_move.depth - 3, analyzed_main_move.depth):
                        if d <= 0:
                            continue
                        mv = self._move_results_by_depth.get(d)
                        if mv and mv.score is not None:
                            trend.append(mv.score)
                    if analyzed_main_move.score is not None:
                        trend.append(analyzed_main_move.score)
                    if trend:
                        ai_meta["scoreTrend"] = trend
            except Exception:
                pass
            ctx = "preview" if analyzed_main_move.context == "preview" else "mainline"
            comment = await self.commenting_service.generate_comment(
                analyzed_main_move, previous_move, pvs_list_of_list_of_moves, context=ctx
            )
            if comment:
                logger.info(f"Sending opening comment for move {comment.moveId}: {comment.text}")
                await send_ws_message(
                    ws,
                    ServerMessageType.COMMENT_UPDATE,
                    comment.model_dump(),
                )
        except Exception as e:
            logger.error(f"Error generating comment: {e}")

    async def _handle_full_game_analysis(self, ws: WebSocket):
        """Handles full game analysis with progress updates"""
        if self._analysis_in_progress:
            await send_ws_message(
                ws, ServerMessageType.ERROR, {"message": "Analysis already in progress"}
            )
            return

        try:
            self._analysis_in_progress = True
            logger.info(f"Starting full game analysis for session {self.session_id}")

            # Define progress callback that sends WebSocket updates
            async def progress_callback(
                current_move: int, total_moves: int, phase: str, move_san: str
            ):
                percentage = (current_move / total_moves) * 100

                progress_payload = AnalysisProgressPayload(
                    current_move=current_move,
                    total_moves=total_moves,
                    percentage=percentage,
                    current_phase=phase,
                    current_move_san=move_san,
                )

                await send_ws_message(
                    ws,
                    ServerMessageType.ANALYSIS_PROGRESS,
                    progress_payload.model_dump(),
                )

            analyzed_moves, all_pvs = (
                await self.analysis_retriever.get_full_game_analysis_with_progress(
                    progress_callback=progress_callback
                )
            )

            # Convert the results to serializable format
            serialized_moves = [move.model_dump() for move in analyzed_moves]
            serialized_pvs = {}
            for move_idx, pvs_list in all_pvs.items():
                serialized_pvs[move_idx] = [
                    [pv_move.model_dump() for pv_move in pv_sequence]
                    for pv_sequence in pvs_list
                ]

            # Send final result
            complete_payload = FullAnalysisCompletePayload(
                moves=serialized_moves, pvs=serialized_pvs
            )

            await send_ws_message(
                ws,
                ServerMessageType.FULL_ANALYSIS_COMPLETE,
                complete_payload.model_dump(),
            )

            # Emit comments for all final-stage moves
            try:
                # Persist the whole game results into the session for downstream consumers
                try:
                    for idx, move in enumerate(analyzed_moves):
                        self._move_results_by_depth[move.depth] = move
                        if idx in all_pvs:
                            self._pvs_by_depth[move.depth] = all_pvs[idx]
                except Exception:
                    pass

                analyzed_moves_with_scores = []
                for idx, move in enumerate(analyzed_moves):
                    if getattr(move, "analysisStage", None) == "final":
                        # Augment trace with mainline UCI sequence for ECO
                        try:
                            seq = [m.move for m in analyzed_moves[:idx+1] if m.move]
                            if seq:
                                if not move.trace:
                                    move.trace = {}
                                move.trace["mainlineUci"] = seq
                        except Exception as e:
                            logger.error(f"Error augmenting trace for full analysis: {e}")

                        previous_move = analyzed_moves_with_scores[-1] if analyzed_moves_with_scores else None
                        comment = await self.commenting_service.generate_comment(
                            move,
                            previous_move,
                            all_pvs.get(idx),
                            context="mainline",
                        )
                        if comment:
                            logger.info(f"Sending opening comment for move {comment.moveId}: {comment.text}")
                            await send_ws_message(
                                ws,
                                ServerMessageType.COMMENT_UPDATE,
                                comment.model_dump(),
                            )
                    analyzed_moves_with_scores.append(move)
            except Exception as e:
                logger.error(f"Error generating batch comments: {e}")

            logger.info(f"Completed full game analysis for session {self.session_id}")

        except Exception as e:
            logger.error(
                f"Error during full analysis for session {self.session_id}: {e}",
                exc_info=True,
            )
            await send_ws_message(
                ws,
                ServerMessageType.ERROR,
                {"message": f"Error during full analysis: {str(e)}"},
            )
        finally:
            self._analysis_in_progress = False

    def _get_analysis_stages(self) -> List[int]:
        """Returns the analysis stages for the engine."""
        return self.analysis_retriever.get_analysis_stages()

    def _get_piece_for_move(
        self, board_before_move: chess.Board, san_move: str
    ) -> str | None:
        """Helper to determine the piece (e.g., wP, bN) that made a move."""
        try:
            move = board_before_move.parse_san(san_move)
            piece = board_before_move.piece_at(move.from_square)
            if piece:
                color_char = "w" if piece.color == chess.WHITE else "b"
                return f"{color_char}{piece.symbol().upper()}"
            return None
        except chess.InvalidMoveError:
            logger.warning(
                f"Could not parse SAN '{san_move}' on board FEN: {board_before_move.fen()}"
            )
            return None
        except Exception as e:
            logger.error(f"Error in _get_piece_for_move for SAN '{san_move}': {e}")
            return None

    async def close_session(self):
        """Placeholder for any session cleanup tasks."""
        logger.info(f"Closing session {self.session_id} (minimal).")
        pass
