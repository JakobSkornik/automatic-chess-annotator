"""Tests for plan tag classification."""

import unittest

import chess

from app.core.commentary.features.plan_extractor import PlanTag, classify_pv_plan


class TestPlanTags(unittest.TestCase):
    def test_pawn_storm_on_wing(self) -> None:
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        tags = classify_pv_plan(board, ["h2h4", "h7h5", "h4h5"])
        self.assertIn(PlanTag.PAWN_STORM.value, tags)

    def test_piece_maneuver_opening(self) -> None:
        board = chess.Board()
        tags = classify_pv_plan(board, ["b1c3", "e7e5", "c3d5", "g8f6", "d5f6"])
        self.assertIn(PlanTag.PIECE_MANEUVER.value, tags)

    def test_rook_invasion(self) -> None:
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        tags = classify_pv_plan(board, ["a1a7"])
        self.assertIn(PlanTag.ROOK_INVASION.value, tags)

    def test_empty_pv(self) -> None:
        board = chess.Board()
        self.assertEqual(classify_pv_plan(board, []), [])


if __name__ == "__main__":
    unittest.main()
