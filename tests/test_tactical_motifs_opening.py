"""Opening moves should not spuriously tag interference."""

from __future__ import annotations

import unittest

import chess

from app.core.commentary.features.tactical_motifs import detect_tactical_motifs


class TestTacticalMotifsOpening(unittest.TestCase):
    def _motifs_after(self, san_seq: list[str]) -> None:
        board = chess.Board()
        for san in san_seq:
            move = board.parse_san(san)
            fb = board.fen()
            board.push(move)
            fa = board.fen()
            motifs = detect_tactical_motifs(
                chess.Board(fb),
                chess.Board(fa),
                move,
                eval_before_cp=0,
                eval_after_cp=0,
            )
            tact_vals = [m.value for m in motifs]
            self.assertNotIn(
                "interference",
                tact_vals,
                msg=f"after {san} in {' '.join(san_seq)} got {tact_vals}",
            )

    def test_first_moves_no_interference(self) -> None:
        self._motifs_after(["e4"])
        self._motifs_after(["e4", "e5"])
        self._motifs_after(["e4", "e5", "Nf3"])
        self._motifs_after(["e4", "e5", "Nf3", "Nc6"])
        self._motifs_after(["e4", "e5", "Nf3", "Nc6", "Nc3"])


if __name__ == "__main__":
    unittest.main()
