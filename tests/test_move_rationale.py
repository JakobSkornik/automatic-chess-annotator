"""MoveRationale builder."""

import unittest

import chess

from app.core.commentary.move_rationale import build_rationale, prompt_projection
from app.models.chess_events import (
    FutureLineDelta,
    MoveEvent,
    MoveEventType,
    MoveQuality,
    PlanComparison,
    TacticalMotif,
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

    def test_prompt_projection_minimal_drops_counterfactual(self) -> None:
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
        r = build_rationale(me, None)
        d = prompt_projection(r, detail="minimal")
        self.assertNotIn("glossary_phrasings", d)
        self.assertNotIn("coach_scratchpad", d)
        self.assertIn("eval_change_cp", d)

    def test_prompt_projection_omits_motif_and_stakes_for_composer(self) -> None:
        b0 = chess.Board()
        b1 = b0.copy()
        b1.push_san("e4")
        b2 = b1.copy()
        b2.push_san("e5")
        me = MoveEvent(
            move_index=1,
            ply=2,
            san="e5",
            uci="e7e5",
            fen_before=b1.fen(),
            fen_after=b2.fen(),
            phase="opening",
            eval_before_cp=20,
            eval_after_cp=18,
            eval_swing_cp=-2,
            move_quality=MoveQuality.BEST,
            event_type=MoveEventType.QUIET,
            tactical_motifs=[TacticalMotif.PIN],
            key_moment_type="critical_decision",
        )
        r = build_rationale(me, None)
        self.assertEqual(r.stakes, "high_stakes_choice")
        self.assertEqual(r.motif, "pin")
        self.assertIsNone(r.risk)
        proj = prompt_projection(r, detail="compact")
        self.assertNotIn("stakes", proj)
        self.assertNotIn("motif", proj)
        self.assertNotIn("risk", proj)


if __name__ == "__main__":
    unittest.main()
