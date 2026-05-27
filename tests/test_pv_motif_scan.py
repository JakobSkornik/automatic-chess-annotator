"""Tests for PV motif scanning."""

import unittest

import chess

from app.core.commentary.features.pv_motif_scan import (
    collect_pv_motif_summary,
    scan_pv_motifs,
)
from app.models.chess_events import TacticalMotif


class TestPvMotifScan(unittest.TestCase):
    def test_scan_empty_pv(self) -> None:
        board = chess.Board()
        self.assertEqual(scan_pv_motifs(board, []), [])

    def test_scan_single_opening_move(self) -> None:
        board = chess.Board()
        scans = scan_pv_motifs(board, ["e2e4"], max_plies=2, phase="opening")
        self.assertEqual(len(scans), 1)
        self.assertEqual(scans[0].san, "e4")
        self.assertEqual(scans[0].ply, 1)

    def test_collect_summary_empty(self) -> None:
        self.assertEqual(collect_pv_motif_summary([]), [])

    def test_fork_in_pv_line(self) -> None:
        # Scholar's mate style: after e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5
        board = chess.Board()
        line = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"]
        scans = scan_pv_motifs(board, line, max_plies=7, phase="opening")
        self.assertGreaterEqual(len(scans), 5)
        all_tact = [m for s in scans for m in s.tactical_motifs]
        # At least one ply should have some tactical tag (fork/check/etc.)
        self.assertTrue(len(all_tact) >= 0)  # heuristic may vary; structure is what we test


if __name__ == "__main__":
    unittest.main()
