"""Detail tier selection for composer/RAG."""

from __future__ import annotations

import unittest

import chess

from app.core.commentary.advanced_comment_service import _detail_level_for_key_moment
from app.models.chess_events import MoveEvent, MoveEventType, MoveQuality


class TestDetailLevelBook(unittest.TestCase):
    def test_opening_best_without_key_moment_is_book(self) -> None:
        b0 = chess.Board()
        b1 = b0.copy()
        b1.push_san("e4")
        me = MoveEvent(
            move_index=0,
            ply=1,
            san="e4",
            uci="e2e4",
            fen_before=b0.fen(),
            fen_after=b1.fen(),
            phase="opening",
            eval_before_cp=0,
            eval_after_cp=16,
            eval_swing_cp=16,
            move_quality=MoveQuality.BEST,
            event_type=MoveEventType.QUIET,
            tactical_motifs=[],
            key_moment_type=None,
            best_move_uci="e2e4",
        )
        self.assertEqual(_detail_level_for_key_moment(me), "book")

    def test_opening_brilliant_not_book(self) -> None:
        b0 = chess.Board()
        b1 = b0.copy()
        b1.push_san("e4")
        me = MoveEvent(
            move_index=0,
            ply=1,
            san="e4",
            uci="e2e4",
            fen_before=b0.fen(),
            fen_after=b1.fen(),
            phase="opening",
            eval_before_cp=0,
            eval_after_cp=16,
            eval_swing_cp=16,
            move_quality=MoveQuality.BEST,
            event_type=MoveEventType.QUIET,
            tactical_motifs=[],
            key_moment_type="brilliant",
            best_move_uci="e2e4",
        )
        self.assertEqual(_detail_level_for_key_moment(me), "full")


if __name__ == "__main__":
    unittest.main()
