"""Move commentary pipeline stage ordering (mocked service)."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock

from app.core.commentary.pipeline.move_pipeline import (
    MoveCommentaryContext,
    MoveCommentaryPipeline,
)
from app.core.commentary.rag_retriever import RAGQuery, RAGResult, RAGRetriever
from app.models.chess_events import (
    GameAnalysisContext,
    MoveCategory,
    MoveEvent,
    MoveEventType,
    MoveQuality,
)


class _StubRAG(RAGRetriever):
    async def retrieve(
        self,
        query: RAGQuery,
        top_k: int = 2,
        *,
        retrieval_debug: dict[str, Any] | None = None,
    ):  # type: ignore[override]
        _ = retrieval_debug
        return [
            RAGResult(
                source="s",
                fen="f",
                annotation_text="note",
                similarity_score=0.5,
            )
        ]


class _FakeCommentService:
    provider_name = "openai"

    def __init__(self) -> None:
        self._rag = _StubRAG()
        self.build_event_llm_input = AsyncMock(
            return_value=(
                "user_prompt",
                [],
                {"composer_named_motifs": [], "rationale": {}},
            )
        )
        self.analyze_and_compose_raw_text = AsyncMock(return_value="final commentary")

    def _rag_retrieve(self, *a: Any, **k: Any) -> Any:
        return self._rag.retrieve(*a, **k)


class TestMoveCommentaryPipeline(unittest.TestCase):
    def test_runs_stages_and_sets_final_text(self) -> None:
        fen_before = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_after = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        me = MoveEvent(
            move_index=0,
            ply=10,
            san="e4",
            uci="e2e4",
            fen_before=fen_before,
            fen_after=fen_after,
            phase="opening",
            eval_before_cp=0,
            eval_after_cp=20,
            move_quality=MoveQuality.GOOD,
            event_type=MoveEventType.QUIET,
            tactical_motifs=[],
            opening_eco="B20",
            pv_san=[],
            move_category=MoveCategory.POSITIONAL,
            key_moment_type="great_move",
        )
        svc = _FakeCommentService()
        ctx = MoveCommentaryContext(
            move_event=me,
            episode=None,
            game_context=GameAnalysisContext(),
            analyzed_row=None,
            service=svc,  # type: ignore[arg-type]
            composer_effort="low",
            key_moment_type="great_move",
        )

        import asyncio

        async def _run() -> None:
            out = await MoveCommentaryPipeline().run(ctx)
            self.assertEqual(out.final_text, "final commentary")
            svc.build_event_llm_input.assert_awaited()
            svc.analyze_and_compose_raw_text.assert_awaited()

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
