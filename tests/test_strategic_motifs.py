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


if __name__ == "__main__":
    unittest.main()
