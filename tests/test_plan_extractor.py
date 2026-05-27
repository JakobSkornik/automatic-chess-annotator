"""Plan extraction from PVs."""

import unittest

from app.core.commentary.features.plan_extractor import build_plan_comparison


class _PM:
    def __init__(self, uci: str) -> None:
        self.move = uci


class TestPlanExtractor(unittest.TestCase):
    def test_played_matches_best_line(self) -> None:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        pvs = [[_PM("e2e4")]]
        pt, bt, ps, bs, pch, bch, ptags, btags = build_plan_comparison(fen, pvs, "e2e4")
        self.assertEqual(ps, bs)
        self.assertTrue(bch)
        self.assertIsInstance(ptags, list)
        self.assertIsInstance(btags, list)


if __name__ == "__main__":
    unittest.main()
