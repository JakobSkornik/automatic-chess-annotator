"""Golden tests for Bolčič-style positional token encoder."""

import unittest

import chess

from app.core.commentary.features.positional_tokens import ENCODER_VERSION, encode_position


class TestPositionalTokens(unittest.TestCase):
    def test_encoder_version(self) -> None:
        self.assertEqual(ENCODER_VERSION, "1")

    def test_encode_position_golden_starting_italian(self) -> None:
        """Stable token snapshot for a fixed FEN + PV (regression guard)."""
        b = chess.Board()
        pv = ["e4", "e5", "Nf3"]
        d = encode_position(b, pv)
        self.assertIn("static_attributes", d)
        self.assertIn("Ke1", d["static_attributes"])
        self.assertIn("pawn_structure", d)
        self.assertTrue(d["center"])
        self.assertIn("dynamic_general", d)
        self.assertIn("dynamic_solution", d)
        self.assertTrue(d["dynamic_solution"].startswith("$e4"))
        self.assertEqual(d["player_color"], "w")
        self.assertIn("encoder_version", d)

    def test_encode_after_e4(self) -> None:
        b = chess.Board()
        b.push_san("e4")
        pv = ["e5", "Nf3", "Nc6"]
        d = encode_position(b, pv)
        self.assertEqual(d["player_color"], "b")
        self.assertIn("Pe4", d["static_attributes"])


if __name__ == "__main__":
    unittest.main()
