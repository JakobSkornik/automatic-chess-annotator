"""MoveRationale builder."""

import unittest

import chess

from app.core.commentary.move_rationale import build_rationale
from app.models.chess_events import (
    FutureLineDelta,
    MoveEvent,
    MoveEventType,
    MoveQuality,
    PlanComparison,
)


class TestMoveRationale(unittest.TestCase):
    def test_build_rationale_has_fields(self) -> None:
        b0 = chess.Board()
        b1 = b0.copy()
        b1.push_san("e4")
        me = MoveEvent(
            move_index=0,
            ply=1,
            san="e4",
            uci="e2e4",
            fen_before=b0.fen(),
            fen_after=b1.fen(),
            phase="opening",
            eval_before_cp=0,
            eval_after_cp=20,
            eval_swing_cp=20,
            move_quality=MoveQuality.GOOD,
            event_type=MoveEventType.QUIET,
            tactical_motifs=[],
            plan_comparison=PlanComparison(
                played_plan_seed="central_play",
                best_plan_seed="central_play",
            ),
        )
        fl = FutureLineDelta(
            eval_gap_cp=40,
            feature_deltas={"king_safety": 0.12},
            best_line_san=["e5", "Nf3"],
        )
        r = build_rationale(me, fl)
        self.assertIn("king_safety", r.future_effect)
        self.assertIsNotNone(r.counterfactual)


if __name__ == "__main__":
    unittest.main()
