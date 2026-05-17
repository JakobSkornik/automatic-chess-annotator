"""Regression tests for clearance / x-ray tactical tagging."""

from __future__ import annotations

import unittest

import chess

from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
from app.models.chess_events import TacticalMotif


class TestClearanceAndXRay(unittest.TestCase):
    def test_corner_rook_slide_not_clearance_or_xray(self) -> None:
        b0 = chess.Board(None)
        for sq, piece in [
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.H1, chess.Piece(chess.ROOK, chess.WHITE)),
            (chess.E8, chess.Piece(chess.KING, chess.BLACK)),
        ]:
            b0.set_piece_at(sq, piece)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("h1g1")
        b1 = b0.copy()
        b1.push(move)
        motifs = detect_tactical_motifs(b0, b1, move, 0, 0)
        vals = [m.value for m in motifs]
        self.assertNotIn("clearance", vals)
        self.assertNotIn("x_ray", vals)

    def test_x_ray_no_tag_when_only_pawn_behind_blocker(self) -> None:
        b0 = chess.Board(None)
        pieces = [
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.H1, chess.Piece(chess.ROOK, chess.WHITE)),
            (chess.G2, chess.Piece(chess.PAWN, chess.WHITE)),
            (chess.G6, chess.Piece(chess.PAWN, chess.BLACK)),
            (chess.G8, chess.Piece(chess.KING, chess.BLACK)),
        ]
        for sq, p in pieces:
            b0.set_piece_at(sq, p)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("h1g1")
        b1 = b0.copy()
        b1.push(move)
        motifs = detect_tactical_motifs(b0, b1, move, 0, 0)
        self.assertNotIn(TacticalMotif.X_RAY, motifs)

    def test_x_ray_tags_major_behind_own_pawn(self) -> None:
        b0 = chess.Board(None)
        pieces = [
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.H1, chess.Piece(chess.ROOK, chess.WHITE)),
            (chess.G2, chess.Piece(chess.PAWN, chess.WHITE)),
            (chess.G7, chess.Piece(chess.QUEEN, chess.BLACK)),
            (chess.G8, chess.Piece(chess.KING, chess.BLACK)),
        ]
        for sq, p in pieces:
            b0.set_piece_at(sq, p)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("h1g1")
        b1 = b0.copy()
        b1.push(move)
        motifs = detect_tactical_motifs(b0, b1, move, 0, 0)
        self.assertIn(TacticalMotif.X_RAY, motifs)

    def test_clearance_when_sliding_piece_unblocks_battery(self) -> None:
        """Queen on d1 masked by rook on d5; rook steps along the file to reveal a new attacked square."""
        b0 = chess.Board(None)
        for sq, pc in [
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.D1, chess.Piece(chess.QUEEN, chess.WHITE)),
            (chess.D5, chess.Piece(chess.ROOK, chess.WHITE)),
            (chess.H8, chess.Piece(chess.KING, chess.BLACK)),
        ]:
            b0.set_piece_at(sq, pc)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("d5d6")
        b1 = b0.copy()
        b1.push(move)
        motifs = detect_tactical_motifs(b0, b1, move, 0, 0)
        self.assertIn(TacticalMotif.CLEARANCE, motifs)


if __name__ == "__main__":
    unittest.main()
