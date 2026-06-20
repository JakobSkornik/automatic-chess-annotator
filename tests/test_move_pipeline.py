"""Move commentary pipeline: with no CommentFacts, the fallback sets final_text."""

from __future__ import annotations

import unittest

from app.core.commentary.pipeline.move_pipeline import (
    MoveCommentaryContext,
    MoveCommentaryPipeline,
)
from app.models.chess_events import (
    GameAnalysisContext,
    MoveCategory,
    MoveEvent,
    MoveEventType,
    MoveQuality,
)


class _FakeCommentService:
    provider_name = "openai"


class TestMoveCommentaryPipeline(unittest.TestCase):
    def test_no_facts_move_falls_back_to_final_text(self) -> None:
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
            move_category=MoveCategory.POSITIONAL,
            key_moment_type="great_move",
        )
        svc = _FakeCommentService()
        ctx = MoveCommentaryContext(
            move_event=me,
            game_context=GameAnalysisContext(),
            analyzed_row=None,
            service=svc,  # type: ignore[arg-type]
            composer_effort="low",
            key_moment_type="great_move",
        )

        import asyncio

        async def _run() -> None:
            out = await MoveCommentaryPipeline().run(ctx)
            # No CommentFacts -> FactsComposeStage yields nothing -> FallbackStage fills it.
            self.assertTrue(out.final_text.strip())
            self.assertIsNotNone(out.fallback_used)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
