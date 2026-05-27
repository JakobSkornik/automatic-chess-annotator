"""Tests for opponent threat detection."""

import unittest

import chess

from app.core.commentary.features.opponent_threats import detect_opponent_threats
from app.models.chess_events import TacticalMotif


class TestOpponentThreats(unittest.TestCase):
    def test_game_over_returns_empty(self) -> None:
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        self.assertTrue(board.is_game_over())
        self.assertEqual(detect_opponent_threats(board), [])

    def test_after_e4_no_crash(self) -> None:
        board = chess.Board()
        board.push_san("e4")
        threats = detect_opponent_threats(board)
        self.assertIsInstance(threats, list)

    def test_pv_reply_used_when_played_matches_best(self) -> None:
        board = chess.Board()
        board.push_san("e4")
        threats = detect_opponent_threats(
            board,
            best_pv_ucis=["e2e4", "e7e5"],
            played_matches_best=True,
        )
        self.assertIsInstance(threats, list)


if __name__ == "__main__":
    unittest.main()
