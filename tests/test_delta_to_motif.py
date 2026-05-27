"""Tests for delta-to-motif mapping."""

import unittest

from app.core.commentary.features.delta_to_motif import infer_motifs_from_deltas
from app.models.chess_events import PvHorizonDiff, StrategicMotif


class TestDeltaToMotif(unittest.TestCase):
    def test_none_diff(self) -> None:
        self.assertEqual(infer_motifs_from_deltas(None), [])

    def test_king_exposure_delta(self) -> None:
        diff = PvHorizonDiff(
            plies=6,
            scalar_deltas={"black.kingExposure": 3.5},
        )
        motifs = infer_motifs_from_deltas(diff)
        self.assertIn(StrategicMotif.WEAK_SQUARE_EXPLOITED, motifs)

    def test_mobility_collapse(self) -> None:
        diff = PvHorizonDiff(
            plies=6,
            scalar_deltas={"white.mobility": -5.0},
        )
        motifs = infer_motifs_from_deltas(diff)
        self.assertIn(StrategicMotif.RESTRICTION, motifs)

    def test_passed_pawn_added(self) -> None:
        diff = PvHorizonDiff(
            plies=8,
            list_deltas={"white.passedPawns": {"added": ["e5"], "removed": []}},
        )
        motifs = infer_motifs_from_deltas(diff, phase="middlegame")
        self.assertIn(StrategicMotif.PASSED_PAWN_MIDDLEGAME, motifs)


if __name__ == "__main__":
    unittest.main()
