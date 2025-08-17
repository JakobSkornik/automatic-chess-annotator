from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any

from app.models.Move import Move, AnalysisStage
from app.models.ws.server_messages import CommentPayload, AiCommentPayload
from app.core.commentary.openings.eco_book import ECOBook
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.ai_comment_service import AICommentService
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
        self._ws_manager = ws_manager

    async def generate_comment(
        self,
        move: Move,
        previous_move: Optional[Move],
        pvs_for_move: Optional[List[List[Move]]],
        context: str,
    ) -> Optional[CommentPayload]:
        # Opening book: if we are still in book, emit opening comment (both stages)
        opening_comment = self._maybe_opening_comment(move)
        if opening_comment:
            return opening_comment

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

                # This is a key moment, so we will generate a comment.
                # We will create a separate task to generate the AI comment
                # and send it back to the frontend when it's ready.
                logger.info(f"Key moment detected for move {move.id}: {key_moment_type}. Triggering AI comment generation.")
                asyncio.create_task(self.generate_ai_comment(move, context))

        return None

    async def generate_ai_comment(self, move: Move, context: str):
        """
        Generates a comment using the AI comment service and sends it to the client.
        """
        # For now, we'll just use the move's trace as the features.
        features = move.trace or {}
        comment_text = await self._ai_comment_service.generate_comment(move, features)

        # Create the payload and send it to the client.
        payload = AiCommentPayload(
            moveId=move.id,
            context=context,
            data={"summary": comment_text, "bullets": []},
        )
        await self._ws_manager.send_comment(payload)

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

