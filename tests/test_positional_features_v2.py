"""Tests for dense positional features (v2 schema) and phase helpers."""

from __future__ import annotations

import unittest

import chess

from app.core.commentary.features.positional_features import compute_hidden_features


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


if __name__ == "__main__":
    unittest.main()
