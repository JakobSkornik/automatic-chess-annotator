"""RAGQuery / build_rag_query wiring."""

import unittest

from app.core.commentary.rag_retriever import build_rag_query
from app.models.chess_events import MoveEvent, MoveEventType, MoveQuality


class TestRAGQuery(unittest.TestCase):
    def test_build_rag_query_includes_fen_and_pv_san(self) -> None:
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
            pv_san=["e5", "Nf3"],
        )
        q = build_rag_query(me, None)
        self.assertEqual(q.fen, me.fen_after)
        self.assertEqual(q.pv_san, ["e5", "Nf3"])


if __name__ == "__main__":
    unittest.main()
