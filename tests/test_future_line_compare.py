"""Tests for future-line comparison (played vs best continuation)."""

import unittest

import chess

from app.core.commentary.features.future_line_compare import (
    compare_played_vs_best_future_lines,
    feature_deltas_between_fens,
)
from app.core.engine.engine_connector import EngineConnector
from app.models.chess_events import FutureLineDelta


class TestFutureLineCompare(unittest.TestCase):
    def test_returns_none_when_best_equals_played(self) -> None:
        b = chess.Board()
        b.push_san("e4")
        fen_after = b.fen()
        self.assertIsNone(
            compare_played_vs_best_future_lines(
                None,  # type: ignore[arg-type]
                chess.STARTING_FEN,
                fen_after,
                "e2e4",
                "e2e4",
            )
        )

    def test_feature_deltas_symmetric_for_identical_fen(self) -> None:
        fen = chess.Board().fen()
        d = feature_deltas_between_fens(fen, fen)
        self.assertTrue(all(abs(v) < 1e-9 for v in d.values()))

    def test_future_line_delta_model(self) -> None:
        d = FutureLineDelta(
            played_leaf_eval_cp=20,
            best_leaf_eval_cp=80,
            eval_gap_cp=60,
            feature_deltas={"white.mobility": 1.5},
            played_targets=["e4"],
            best_targets=["d5"],
            played_line_san=["Nf3"],
            best_line_san=["d5"],
        )
        self.assertEqual(d.eval_gap_cp, 60)


@unittest.skipUnless(
    __import__("os").path.isfile(
        __import__("os").path.abspath(
            __import__("os").path.join(
                __import__("os").path.dirname(__file__),
                "..",
                "stockfish.exe",
            )
        )
    ),
    "stockfish.exe not at repo root",
)
class TestFutureLineCompareIntegration(unittest.TestCase):
    """One integration test with real Stockfish (optional)."""

    def test_compare_after_e4(self) -> None:
        import os

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = os.path.join(root, "stockfish.exe")
        eng = EngineConnector(path)
        try:
            # Played suboptimal 1.h3; engine best from root is usually 1.e4 or 1.d4
            b0 = chess.Board()
            b0.push_san("h3")
            fen_after_played = b0.fen()
            info = eng.analyse(chess.Board(chess.STARTING_FEN), depth=8, multiPv=1)
            if isinstance(info, list) and info:
                info = info[0]
            pv = (info or {}).get("pv") or []
            if not pv:
                self.skipTest("no PV")
            best_uci = pv[0].uci()
            if best_uci == "h2h3":
                self.skipTest("engine agrees with h3")
            delta = compare_played_vs_best_future_lines(
                eng,
                chess.STARTING_FEN,
                fen_after_played,
                best_uci,
                "h2h3",
                depth=10,
                n_plies=4,
            )
            self.assertIsNotNone(delta)
            self.assertIsInstance(delta, FutureLineDelta)
        finally:
            eng.close()


if __name__ == "__main__":
    unittest.main()
