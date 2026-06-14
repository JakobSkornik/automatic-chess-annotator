"""Strategic motif smoke tests."""

import unittest

import chess

from app.core.commentary.features.strategic_motifs import detect_strategic_motifs


class TestStrategicMotifs(unittest.TestCase):
    def test_detect_runs_on_starting_position(self) -> None:
        b0 = chess.Board()
        m = chess.Move.from_uci("e2e4")
        b1 = b0.copy()
        b1.push(m)
        out = detect_strategic_motifs(b0, b1, m, {}, eval_after_cp=20, phase="opening")
        self.assertIsInstance(out, list)

    def test_bad_bishop_not_tagged_for_three_same_color_pawns(self) -> None:
        """Engine-style badBishops=1 with only three same-square-color pawns → skip motif."""
        b_setup = chess.Board(None)
        # Black bishop on g7 with exactly three black pawns on the same square color (dark vs light).
        for sq, pc in [
            (chess.G8, chess.Piece(chess.KING, chess.BLACK)),
            (chess.G7, chess.Piece(chess.BISHOP, chess.BLACK)),
            (chess.A7, chess.Piece(chess.PAWN, chess.BLACK)),
            (chess.C7, chess.Piece(chess.PAWN, chess.BLACK)),
            (chess.E7, chess.Piece(chess.PAWN, chess.BLACK)),
            (chess.E2, chess.Piece(chess.PAWN, chess.WHITE)),
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
        ]:
            b_setup.set_piece_at(sq, pc)
        b_setup.turn = chess.BLACK
        m = chess.Move.from_uci("a7a6")
        b1 = b_setup.copy()
        b1.push(m)
        hf = {"black": {"badBishops": 1}}
        out = detect_strategic_motifs(
            b_setup, b1, m, hf, eval_after_cp=0, phase="opening"
        )
        vals = [x.value for x in out]
        self.assertNotIn("bad_bishop", vals)


if __name__ == "__main__":
    unittest.main()
