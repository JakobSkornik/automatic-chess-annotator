"""LLM debug payload from build_event_llm_input (no API key required)."""

import unittest

from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.rag_retriever import RAGQuery, RAGResult, RAGRetriever
from app.models.chess_events import (
    GameAnalysisContext,
    MoveCategory,
    MoveEvent,
    MoveEventType,
    MoveQuality,
)

_LONG_STUB_NOTE = (
    "White develops naturally while Black fianchettoes the king bishop. "
    "Both sides contest the center with pawns and minor pieces. "
    "The middlegame brings opposite-side castling and mutual threats. "
    "Tactical motifs appear along open files and weak squares near the king. "
    "Eventually simplifications favor the side with better coordination. "
    "stub master note marker for debug test."
)


class _StubRAG(RAGRetriever):
    async def retrieve(
        self,
        query: RAGQuery,
        top_k: int = 2,
        *,
        retrieval_debug=None,
    ):  # type: ignore[override]
        return [
            RAGResult(
                source="stub.pgn",
                fen="fen",
                annotation_text=_LONG_STUB_NOTE,
                similarity_score=0.9,
            )
        ]


class TestLlmDebug(unittest.TestCase):
    def test_build_event_llm_input_debug_dict(self) -> None:
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
            pv_san=["e5", "Nf3"],
            move_category=MoveCategory.POSITIONAL,
            key_moment_type="great_move",
        )
        for provider_key in ("openai", "anthropic"):
            svc = AdvancedCommentService(rag_retriever=_StubRAG(), provider_key=provider_key)
            self.assertEqual(svc.provider_name, provider_key)
        svc = AdvancedCommentService(rag_retriever=_StubRAG(), provider_key="openai")
        ctx = GameAnalysisContext()

        async def _run() -> None:
            structured_text, _rag, debug = await svc.build_event_llm_input(
                me, None, ctx, analyzed_row=None
            )
            self.assertIn("MOVE_RATIONALE_JSON", structured_text)
            self.assertIn("===DYNAMIC===", structured_text)
            self.assertIn("MASTER ANNOTATIONS", structured_text)
            self.assertIn("stub master note marker", structured_text)
            rationale_pos = structured_text.rfind("MOVE_RATIONALE_JSON")
            master_pos = structured_text.rfind("MASTER ANNOTATIONS")
            self.assertGreater(master_pos, rationale_pos)
            self.assertIn("rag_query", debug)
            self.assertIn("rationale", debug)
            self.assertIn("system_prompts", debug)
            self.assertIn("user_text", debug)
            self.assertIn("passes", debug)
            self.assertEqual(debug["user_text"], structured_text)
            self.assertIsNone(debug.get("token_usage_total"))
            self.assertIsInstance(debug["system_prompts"], list)
            self.assertTrue(len(debug["system_prompts"]) >= 1)
            self.assertIn("MOVE_RATIONALE_JSON", debug["user_text"])
            self.assertIn("immediate_effect", debug["user_text"])

        import asyncio

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
