from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any

from app.models.Move import Move, AnalysisStage
from app.models.ws.server_messages import CommentPayload, AiCommentPayload
from app.core.commentary.openings.eco_book import ECOBook
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.ai_comment_service import AICommentService
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.websocket.ws_manager import WebSocketManager

logger = logging.getLogger(__name__)


class CommentingService:
    """Generates comments for moves based on analysis artifacts.

    This is a minimal scaffold: it uses score and phase to emit a basic comment.
    Later, it will leverage a position graph, feature extractors, and templates.
    """

    def __init__(self, ws_manager: WebSocketManager) -> None:
        self._version: int = 1
        self._eco = ECOBook()
        self._key_moment_detector = KeyMomentDetector()
        self._ai_comment_service = AICommentService()
        self._advanced_service = AdvancedCommentService()
        self._ws_manager = ws_manager
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

        # If there is no opening comment, check for a key moment.
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

        # Always attempt AI comment generation for quiescent moves (generation method will skip noisy moves)
        asyncio.create_task(self.generate_ai_comment(move, context, pvs_for_move, previous_move))

        return None

    async def generate_ai_comment(self, move: Move, context: str, pvs_for_move: Optional[List[List[Move]]] = None, previous_move: Optional[Move] = None):
        """
        Generates a comment using the AI comment service and sends it to the client.
        """
        # Only comment on quiescent moves (no capture, no check) to avoid horizon noise
        try:
            san = (move.trace or {}).get("san", "") if isinstance(move.trace, dict) else ""
            is_capture = "x" in san
            is_check = "+" in san or "#" in san
            if is_capture or is_check:
                return
        except Exception:
            pass

        # Opening context to include in input and allow AI to mention
        opening_ctx = {}
        try:
            if isinstance(move.trace, dict):
                opening_ctx = {"eco": move.trace.get("openingCode"), "name": move.trace.get("openingName"), "variation": None}
                # Try to enrich via ECOBook using mainline UCI if available
                seq = move.trace.get("mainlineUci") if isinstance(move.trace.get("mainlineUci"), list) else []
                if seq:
                    info = self._eco.match(seq)
                    if info:
                        opening_ctx["eco"] = getattr(info, "code", opening_ctx.get("eco"))
                        # Build label with variation if present
                        nm = getattr(info, "name", None)
                        var = getattr(info, "variation", None)
                        if nm:
                            opening_ctx["name"] = nm
                        if var:
                            opening_ctx["variation"] = var
        except Exception:
            opening_ctx = {}
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

        # Build compact, PV-aware input for two-step pipeline
        compact = self._advanced_service.build_compact_input(
            move, previous_move=previous_move, pvs_for_move=pvs_for_move, opening=opening_ctx
        )
        comment_text = await self._advanced_service.analyze_and_compose(
            compact,
            model=self._model_params.get("model"),
            effort=self._model_params.get("effort"),
            temperature=self._model_params.get("temperature"),
            max_tokens=self._model_params.get("maxTokens"),
        )

        # Create the payload and send it to the client.
        payload = AiCommentPayload(
            moveId=move.id,
            context=context,
            data={"summary": comment_text, "bullets": []},
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

