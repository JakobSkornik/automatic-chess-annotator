"""Regression tests for the settled-material-deficit sacrifice core.

Covers the fix for the material-comparison bug described in the code audit:
TacticalMotif.SACRIFICE / POSITIONAL_PAWN_SAC / DECOY used to diff the
mover's own material one ply apart on the SAME half-move, which can never
fire (a single legal move can't reduce the mover's own material). The fix
walks forward past the opponent's reply to a settled/quiescent leaf
(sacrifice_core.settled_material_deficit) before comparing.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import chess

from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
from app.core.commentary.features.tactical_motifs.sacrifice_core import (
    is_sacrifice,
    settled_material_deficit,
)
from app.core.engine.analysis.key_moments import _eval_clearly_winning, _is_sacrifice
from app.models.chess_events import TacticalMotif


class TestSettledMaterialDeficitCore(unittest.TestCase):
    def test_real_sacrifice_persists_and_is_flagged(self) -> None:
        """Qxh7+ Kxh7: White gives up the whole queen for a pawn — a real,
        persistent sacrifice. With a merely "not clearly losing" eval after
        (0cp), the general-purpose predicate must fire."""
        b0 = chess.Board(None)
        for sq, piece in [
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.H5, chess.Piece(chess.QUEEN, chess.WHITE)),
            (chess.G8, chess.Piece(chess.KING, chess.BLACK)),
            (chess.H7, chess.Piece(chess.PAWN, chess.BLACK)),
        ]:
            b0.set_piece_at(sq, piece)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("h5h7")
        self.assertIn(move, b0.legal_moves)
        b1 = b0.copy()
        b1.push(move)

        deficit = settled_material_deficit(b0, b1, chess.WHITE)
        self.assertGreaterEqual(deficit, 700)  # ~queen for a pawn
        self.assertTrue(is_sacrifice(b0, b1, chess.WHITE, 0))

        motifs = detect_tactical_motifs(b0, b1, move, 0, 0)
        self.assertIn(TacticalMotif.SACRIFICE, motifs)

    def test_ordinary_trade_settles_to_equality_not_a_sacrifice(self) -> None:
        """Rxd8: an even rook trade. Material returns to equality at the
        settled leaf, so this must NOT be tagged a sacrifice."""
        b0 = chess.Board(None)
        for sq, piece in [
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.D1, chess.Piece(chess.ROOK, chess.WHITE)),
            (chess.E8, chess.Piece(chess.KING, chess.BLACK)),
            (chess.D8, chess.Piece(chess.ROOK, chess.BLACK)),
        ]:
            b0.set_piece_at(sq, piece)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("d1d8")
        self.assertIn(move, b0.legal_moves)
        b1 = b0.copy()
        b1.push(move)

        deficit = settled_material_deficit(b0, b1, chess.WHITE)
        self.assertEqual(deficit, 0)
        self.assertFalse(is_sacrifice(b0, b1, chess.WHITE, 0))

        motifs = detect_tactical_motifs(b0, b1, move, 0, 0)
        self.assertNotIn(TacticalMotif.SACRIFICE, motifs)

    def test_plain_blunder_that_stays_lost_is_not_a_sacrifice(self) -> None:
        """The queen wanders onto a square a rook takes for free, and the
        resulting eval is clearly lost (no compensation) — an ordinary
        blunder, not a sacrifice, despite the large persistent deficit."""
        b0 = chess.Board(None)
        for sq, piece in [
            (chess.E1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.A4, chess.Piece(chess.QUEEN, chess.WHITE)),
            (chess.E8, chess.Piece(chess.KING, chess.BLACK)),
            (chess.A8, chess.Piece(chess.ROOK, chess.BLACK)),
        ]:
            b0.set_piece_at(sq, piece)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("a4a7")
        self.assertIn(move, b0.legal_moves)
        b1 = b0.copy()
        b1.push(move)

        deficit = settled_material_deficit(b0, b1, chess.WHITE)
        self.assertGreaterEqual(deficit, 800)  # queen dropped
        # Clearly losing afterwards (mover-POV eval well below the floor):
        self.assertFalse(is_sacrifice(b0, b1, chess.WHITE, -900))

        motifs = detect_tactical_motifs(b0, b1, move, 0, -900)
        self.assertNotIn(TacticalMotif.SACRIFICE, motifs)

    def test_positional_pawn_sac_narrower_than_sacrifice(self) -> None:
        """A lone-pawn persistent deficit (not a bigger sacrifice) tags
        POSITIONAL_PAWN_SAC, not SACRIFICE."""
        b0 = chess.Board(None)
        for sq, piece in [
            (chess.A1, chess.Piece(chess.KING, chess.WHITE)),
            (chess.G5, chess.Piece(chess.PAWN, chess.WHITE)),
            (chess.H7, chess.Piece(chess.KING, chess.BLACK)),
        ]:
            b0.set_piece_at(sq, piece)
        b0.turn = chess.WHITE
        move = chess.Move.from_uci("g5g6")
        self.assertIn(move, b0.legal_moves)
        b1 = b0.copy()
        b1.push(move)

        deficit = settled_material_deficit(b0, b1, chess.WHITE)
        self.assertEqual(deficit, 100)

        motifs = detect_tactical_motifs(b0, b1, move, 0, 0)
        self.assertIn(TacticalMotif.POSITIONAL_PAWN_SAC, motifs)
        self.assertNotIn(TacticalMotif.SACRIFICE, motifs)


class TestBrilliancyPathUnaffected(unittest.TestCase):
    """key_moments._is_sacrifice now calls the shared persistent_deficit
    core, but must behave exactly as before."""

    @staticmethod
    def _facts(series: list[int], mover: str, eval_cp: int | None) -> SimpleNamespace:
        display_line = SimpleNamespace(feature_series={"MATERIAL_BALANCE": series})
        return SimpleNamespace(
            display_line=display_line, mover=mover, eval_cp=eval_cp, eval_mate=None
        )

    def test_persistent_deficit_that_stays_winning_is_brilliant_sac(self) -> None:
        facts = self._facts([0, -300, -300, -300], "White", 150)
        self.assertTrue(_is_sacrifice(facts))
        self.assertTrue(_eval_clearly_winning(facts))

    def test_transient_dip_regained_is_not_a_sacrifice(self) -> None:
        facts = self._facts([0, -300, -300, 0], "White", 20)
        self.assertFalse(_is_sacrifice(facts))


if __name__ == "__main__":
    unittest.main()
