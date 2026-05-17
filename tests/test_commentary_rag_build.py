"""Golden checks for commentary user-blob composition (no API keys)."""

from __future__ import annotations

import unittest

from app.core.commentary.advanced_comment_service import (
    AdvancedCommentService,
    COMPOSER_OUTPUT_SCHEMA,
)
from app.core.commentary.rag_retriever import RAGQuery, RAGResult, RAGRetriever
from app.models.GameJson import AnalysisInfo, GameJson, GameMetadata, GameMove, RagRef
from app.models.chess_events import GameAnalysisContext, MoveCategory, MoveEvent, MoveEventType, MoveQuality


_LONG_MASTER_NOTE = (
    "White develops naturally while Black fianchettoes the king bishop. "
    "Both sides contest the center with pawns and minor pieces. "
    "The middlegame brings opposite-side castling and mutual threats. "
    "Tactical motifs appear along open files and weak squares near the king. "
    "Eventually simplifications favor the side with better coordination. "
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
                source="gm.pgn",
                fen="8/8/8/8/8/8/8/8 b - -",
                annotation_text=_LONG_MASTER_NOTE + "Paraphrase-able master idea.",
                similarity_score=0.77,
                relevance_tags={
                    "phase": "middlegame",
                    "opening_name": "Four Knights Scotch",
                    "opening_eco": "C47",
                    "material_signature": "equal",
                    "opening": "",
                },
            )
        ]


class TestCommentaryRagGolden(unittest.TestCase):
    def test_rag_block_is_last_dynamic_section_with_header(self) -> None:
        fen_before = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_after = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        me = MoveEvent(
            move_index=0,
            ply=10,
            san="e4",
            uci="e2e4",
            fen_before=fen_before,
            fen_after=fen_after,
            phase="middlegame",
            eval_before_cp=0,
            eval_after_cp=18,
            move_quality=MoveQuality.GOOD,
            event_type=MoveEventType.QUIET,
            tactical_motifs=[],
            opening_eco="C47",
            opening_name="Four Knights Scotch",
            pv_san=["Nf6", "Nf3"],
            move_category=MoveCategory.BOOK,
            key_moment_type="critical_decision",
        )
        svc = AdvancedCommentService(rag_retriever=_StubRAG(), provider_key="openai")

        async def _run() -> None:
            blob, refs, dbg = await svc.build_event_llm_input(me, None, GameAnalysisContext())
            self.assertGreater(len(refs), 0)
            self.assertIn("MASTER ANNOTATIONS", blob)
            self.assertIn("Paraphrase-able master idea.", blob)
            self.assertNotIn(
                "inspiration, not evaluation",
                blob.lower(),
            )
            rationale_pos = blob.rfind("MOVE_RATIONALE_JSON")
            master_pos = blob.rfind("MASTER ANNOTATIONS")
            fen_last = blob.rfind("FEN after")
            self.assertGreater(master_pos, rationale_pos)
            self.assertGreater(master_pos, fen_last)
            tags = refs[0].relevance_tags
            self.assertEqual(tags.get("opening_name"), "Four Knights Scotch")
            self.assertIn("rag_hit_count", dbg)
            self.assertEqual(dbg.get("rag_hit_count"), 1)

        import asyncio

        asyncio.run(_run())

    def test_composer_schema_contains_rag_and_alternative_fields(self) -> None:
        props = set(COMPOSER_OUTPUT_SCHEMA["properties"].keys())
        for k in ("better_alternative", "rag_idea_used", "rag_applied", "named_motifs", "text"):
            self.assertIn(k, props)

    def test_game_json_rag_refs_round_trip(self) -> None:
        meta = GameMetadata(
            id="t",
            white="w",
            black="b",
            result="*",
            opening_eco=None,
            opening=None,
        )
        mv = GameMove(
            mn=1,
            color="w",
            san="e4",
            uci="e2e4",
            fen=_START_FEN,
            rag_refs=[
                RagRef(
                    source="s",
                    text="snippet",
                    score=0.4,
                    opening_eco="B20",
                    opening_name="Sicilian",
                    material_signature="equal",
                )
            ],
        )
        g = GameJson(
            metadata=meta,
            moves=[mv],
            analysis_info=AnalysisInfo(engine="s", depth=14, multipv=2, timestamp=1.0),
        )
        g2 = GameJson.model_validate_json(g.model_dump_json())
        self.assertEqual(len(g2.moves[0].rag_refs), 1)
        self.assertEqual(g2.moves[0].rag_refs[0].opening_name, "Sicilian")

_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


if __name__ == "__main__":
    unittest.main()
