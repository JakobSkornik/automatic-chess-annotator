from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any

import chess
from chess.pgn import Game

from app.core.commentary.annotation_tokens import resolve_tokens_for_comment

from app.models.Move import Move, AnalysisStage
from app.models.chess_events import GameAnalysisContext
from app.core.commentary.event_extractor import (
    ChessEventExtractor,
    build_analyzed_row_from_interactive,
)
from app.models.ws.server_messages import CommentPayload, AiCommentPayload
from app.core.commentary.openings.eco_book import ECOBook
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.ai_comment_service import AICommentService
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.chroma_rag_retriever import get_default_retriever
from app.core.commentary.rag_retriever import rag_results_to_ws_refs
from app.core.commentary.narrative_tracker import NarrativeTracker
from app.core.websocket.ws_manager import WebSocketManager

logger = logging.getLogger(__name__)


class CommentingService:
    """Generates comments for moves based on analysis artifacts.

    This is a minimal scaffold: it uses score and phase to emit a basic comment.
    Later, it will leverage a position graph, feature extractors, and templates.
    """

    def __init__(
        self,
        ws_manager: WebSocketManager,
        pgn_game: Optional[Game] = None,
    ) -> None:
        self._version: int = 1
        self._eco = ECOBook()
        self._key_moment_detector = KeyMomentDetector()
        self._ai_comment_service = AICommentService()
        self._advanced_service = AdvancedCommentService(rag_retriever=get_default_retriever())
        self._narrative = NarrativeTracker()
        self._ws_manager = ws_manager
        self._pgn_game: Optional[Game] = pgn_game
        self._model_params = {"model": "gpt-5-mini", "effort": "low", "temperature": 0.2, "maxTokens": 120}

    def update_model_params(self, params: dict) -> None:
        self._model_params.update(params or {})

    async def generate_comment(
        self,
        move: Move,
        previous_move: Optional[Move],
        pvs_for_move: Optional[List[List[Move]]],
        context: str,
    ) -> Optional[CommentPayload]:
        # Do not emit opening text; pass opening context into AI input instead.

        # Detect key moments
        key_moment_type: Optional[str] = None
        prev_score = previous_move.score if previous_move else "N/A"
        logger.info(
            f"Checking for key moment on move {move.depth}. "
            f"Move Score: {move.score}, Previous Move Score: {prev_score}"
        )
        if move.score is not None and previous_move and previous_move.score is not None:
            key_moment_type = self._key_moment_detector.detect(move, previous_move, pvs_for_move)
            if key_moment_type:
                if not move.hiddenFeatures:
                    move.hiddenFeatures = {}
                move.hiddenFeatures["keyMomentType"] = key_moment_type

        # Update the narrative tracker with this move
        opening_ctx = self._extract_opening_context(move)
        self._narrative.update(
            ply=move.depth,
            score=move.score,
            key_moment_type=key_moment_type,
            hidden_features=move.hiddenFeatures,
            opening=opening_ctx,
        )

        # Generate AI comment (with narrative context and key moment type)
        asyncio.create_task(
            self.generate_ai_comment(move, context, pvs_for_move, previous_move, key_moment_type)
        )

        return None

    def _extract_opening_context(self, move: Move) -> Dict[str, Any]:
        """Extract opening ECO/name/variation from a move's trace data."""
        opening_ctx: Dict[str, Any] = {}
        try:
            if isinstance(move.trace, dict):
                opening_ctx = {
                    "eco": move.trace.get("openingCode"),
                    "name": move.trace.get("openingName"),
                    "variation": None,
                }
                seq = move.trace.get("mainlineUci") if isinstance(move.trace.get("mainlineUci"), list) else []
                if seq:
                    info = self._eco.match(seq)
                    if info:
                        opening_ctx["eco"] = getattr(info, "code", opening_ctx.get("eco"))
                        nm = getattr(info, "name", None)
                        var = getattr(info, "variation", None)
                        if nm:
                            opening_ctx["name"] = nm
                        if var:
                            opening_ctx["variation"] = var
        except Exception:
            opening_ctx = {}
        return opening_ctx

    async def generate_ai_comment(
        self,
        move: Move,
        context: str,
        pvs_for_move: Optional[List[List[Move]]] = None,
        previous_move: Optional[Move] = None,
        key_moment_type: Optional[str] = None,
    ):
        """
        Generates a comment using the AI comment service and sends it to the client.
        """
        # Previously, captures and checks were skipped to avoid horizon noise.
        # This filter is removed: captures and checks are often the most critical
        # and interesting moves (sacrificial attacks, decisive combinations) and
        # deserve strategic commentary.

        opening_ctx = self._extract_opening_context(move)

        # Emit start status
        try:
            from app.models.ws.server_messages import AiGenerationStatusPayload
            import time as _t
            await self._ws_manager.send_generation_status(
                AiGenerationStatusPayload(
                    moveId=move.id,
                    context=context,
                    status="start",
                    startedAt=_t.time(),
                    model=str(self._model_params.get("model")),
                    effort=str(self._model_params.get("effort")),
                )
            )
        except Exception:
            pass

        narrative_context = self._narrative.get_narrative_context()
        comment_text = ""
        rag_refs_ws: List[Dict[str, Any]] = []
        use_event_pipeline = (
            self._pgn_game is not None
            and move.depth
            and pvs_for_move is not None
        )
        if use_event_pipeline:
            try:
                move_idx = max(0, int(move.depth) - 1)
                row = build_analyzed_row_from_interactive(
                    self._pgn_game, move_idx, move, pvs_for_move
                )
                extractor = ChessEventExtractor()
                events = extractor.extract_events(
                    self._pgn_game,
                    [row],
                    previous_engine_move=previous_move,
                )
                if events:
                    me = events[0]
                    gctx = GameAnalysisContext(
                        metadata={},
                        move_events=events,
                        episodes=[],
                        critical_moments=[me] if me.is_critical else [],
                    )
                    if me.is_critical:
                        comment_text, rag_results = await self._advanced_service.analyze_and_compose_event(
                            me,
                            None,
                            gctx,
                            model=self._model_params.get("model"),
                            effort=self._model_params.get("effort"),
                            key_moment_type=key_moment_type,
                        )
                        rag_refs_ws = rag_results_to_ws_refs(rag_results)
            except Exception as e:
                logger.error(f"Event pipeline comment failed, falling back: {e}")
                comment_text = ""

        if not comment_text:
            compact = self._advanced_service.build_compact_input(
                move,
                previous_move=previous_move,
                pvs_for_move=pvs_for_move,
                opening=opening_ctx,
                narrative_context=narrative_context,
                key_moment_type=key_moment_type,
            )
            comment_text = await self._advanced_service.analyze_and_compose(
                compact,
                model=self._model_params.get("model"),
                effort=self._model_params.get("effort"),
                temperature=self._model_params.get("temperature"),
                max_tokens=self._model_params.get("maxTokens"),
                key_moment_type=key_moment_type,
            )

        fen_after = move.position
        fen_before = (
            previous_move.position
            if previous_move
            else chess.Board().fen()
        )
        resolved_tokens = resolve_tokens_for_comment(comment_text, fen_before, fen_after)

        # Create the payload and send it to the client.
        payload = AiCommentPayload(
            moveId=move.id,
            context=context,
            data={
                "summary": comment_text,
                "commentary": comment_text,
                "bullets": [],
                "resolved_tokens": resolved_tokens,
                "rag_refs": rag_refs_ws,
            },
        )
        await self._ws_manager.send_comment(payload)

        # Emit end status
        try:
            from app.models.ws.server_messages import AiGenerationStatusPayload
            import time as _t
            await self._ws_manager.send_generation_status(
                AiGenerationStatusPayload(
                    moveId=move.id,
                    context=context,
                    status="end",
                    endedAt=_t.time(),
                    model=str(self._model_params.get("model")),
                    effort=str(self._model_params.get("effort")),
                )
            )
        except Exception:
            pass

    @staticmethod
    def _is_white_move(depth: int) -> bool:
        # depth is 1-based where odd => White, even => Black
        return depth % 2 == 1

    def _maybe_opening_comment(self, move: Move) -> Optional[CommentPayload]:
        # Rebuild mainline up to current depth as UCI list to match ECO
        # The Move.move field contains UCI for this ply; we need the prefix sequence.
        # For now, when depth <= 10 plies, we try a crude accumulation using move.trace if present.
        # In a later pass, we should feed the full game sequence to the service.
        try:
            # Try to read sequence from trace if available
            seq: List[str] = []
            if move.trace and isinstance(move.trace, dict):
                seq = move.trace.get("mainlineUci", []) or []
            # If we have at least the first few moves, attempt match; do not fallback to single-move
            if seq:
                info = self._eco.match(seq)
                if info:
                    name = info.name if not info.variation else f"{info.name}: {info.variation}"
                    text = f"Opening: {info.code} {name}."
                    return CommentPayload(
                        moveId=move.id,
                        context="mainline",
                        text=text,
                        featuresUsed=["opening"],
                        hiddenFeatures=None,
                        analysisVersion=move.analysisVersion,
                    )
        except Exception:
            pass
        return None

