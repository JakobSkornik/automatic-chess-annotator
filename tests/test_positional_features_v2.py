"""Tests for dense positional features (v2 schema), phase helpers, and PV horizon diff."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import chess

from app.core.commentary.features.positional_features import compute_hidden_features
from app.core.commentary.features.pv_horizon_diff import compute_pv_horizon_diff
from app.core.commentary.features.rag_phase_features import (
    classify_rag_phase,
    in_opening_book,
)


class TestPositionalFeaturesV2(unittest.TestCase):
    def test_doubled_pawns_square_lists(self) -> None:
        # Two white pawns on the d-file (doubled).
        b = chess.Board("8/8/8/8/3P4/3P4/8/4K2k w - - 0 1")
        f = compute_hidden_features(b)
        w = f["white"]
        dp = w["doubledPawns"]
        self.assertIsInstance(dp, list)
        self.assertTrue({"d3", "d4"}.issubset(set(dp)))

    def test_passed_pawns_dict_records(self) -> None:
        b = chess.Board("8/8/8/8/P7/8/5K1k/8 w - - 0 1")
        f = compute_hidden_features(b)
        passed = f["white"]["passedPawns"]
        self.assertTrue(
            any(isinstance(x, dict) and x.get("sq") == "a4" for x in passed)
        )

    def test_material_skeleton_survives_dense(self) -> None:
        b = chess.Board()
        f = compute_hidden_features(b)
        self.assertIn("material", f)
        self.assertIn("diff", f["material"])

    def test_board_overlay_open_files(self) -> None:
        b = chess.Board()
        f = compute_hidden_features(b)
        bo = f["boardOverlay"]
        # Empty lists are dropped from overlay (dense output).
        self.assertTrue(
            bo.get("openFiles") is None or isinstance(bo.get("openFiles"), list)
        )

    def test_in_opening_book_empty_prefix(self) -> None:
        hit, n = in_opening_book([])
        self.assertFalse(hit)
        self.assertEqual(n, 0)

    def test_classify_rag_phase_with_book_params_out_of_book(self) -> None:
        b = chess.Board()
        # Pretend only 2 plies matched but 4 played => middlegame classification path
        rag = classify_rag_phase(b, opening_matched_ply=2, uci_plies_played=4)
        self.assertEqual(rag, "middlegame")

    def test_pv_horizon_diff_returns_none_when_pv_short(self) -> None:
        board = chess.Board()
        mv = list(board.legal_moves)[:3]
        self.assertLess(len(mv), 4)

        sc = MagicMock()
        sc.white.return_value.score.return_value = 15
        info = {"score": sc, "pv": mv}

        eng = MagicMock()
        eng.analyse.return_value = info

        out = compute_pv_horizon_diff(eng, board.fen(), plies=10, depth=8)
        self.assertIsNone(out)
        eng.analyse.assert_called()


if __name__ == "__main__":
    unittest.main()
