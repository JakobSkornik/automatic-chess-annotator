"""Perspective and decisive-interval regressions for move-quality rules."""

import chess
import pytest

from app.core.commentary.event_extractor import _move_quality_for
from app.core.commentary.features.move_category import classify_move_event
from app.core.commentary.key_moment_detector import KeyMomentDetector
from app.core.commentary.rules.verdicts import verdict_for_transition
from app.models.Move import Move
from app.models.chess_events import MoveCategory, MoveEvent, MoveEventType, MoveQuality


@pytest.mark.parametrize("sign", [1, -1])
def test_decisive_sign_flip_is_still_a_blunder(sign):
    assert _move_quality_for(600, sign * 300, -sign * 300, False) == MoveQuality.BLUNDER


@pytest.mark.parametrize("sign", [1, -1])
def test_same_side_decisive_interval_policy_is_preserved(sign):
    assert _move_quality_for(150, sign * 500, sign * 350, False) == MoveQuality.GOOD


@pytest.mark.parametrize("sign,mover", [(1, "White"), (-1, "Black")])
def test_verdict_boundary_agrees_with_absolute_advantage_brackets(sign, mover):
    # +/-25 cp is a slight advantage, not part of the equality interval.
    assert "slight advantage" in verdict_for_transition(-100 * sign, 25 * sign, None, mover)
    assert "balance" not in verdict_for_transition(-100 * sign, -25 * sign, None, mover)
    assert "equality" not in verdict_for_transition(100 * sign, 25 * sign, None, mover)


@pytest.mark.parametrize("black", [False, True])
def test_key_moment_keeps_winning_to_losing_blunder(black):
    before = chess.Board()
    if black:
        before.push_san("Nf3")
    after = before.copy()
    move = "g8f6" if black else "g1f3"
    after.push_uci(move)
    sign = -1 if black else 1
    previous = Move(id=1, position=before.fen(), move="g1f3", depth=1,
                    context="mainline", isAnalyzed=True, piece="wN", score=sign * 300)
    current = Move(id=2, position=after.fen(), move=move, depth=2,
                   context="mainline", isAnalyzed=True, piece="bN" if black else "wN",
                   score=-sign * 300)
    assert "blunder" in KeyMomentDetector()._score_based(current, previous, None)


@pytest.mark.parametrize("black", [False, True])
@pytest.mark.parametrize("loss,expected", [(0, MoveCategory.DEFENSIVE), (30, MoveCategory.DEFENSIVE),
                                           (-50, MoveCategory.DEFENSIVE), (80, MoveCategory.POSITIONAL)])
def test_defensive_category_uses_mover_perspective(black, loss, expected):
    before = chess.Board()
    if black:
        before.push_san("Nf3")
    after = before.copy()
    uci = "g8f6" if black else "g1f3"
    san = after.san(chess.Move.from_uci(uci))
    after.push_uci(uci)
    sign = -1 if black else 1
    event = MoveEvent(
        move_index=0, ply=2 if black else 1, san=san, uci=uci,
        fen_before=before.fen(), fen_after=after.fen(), phase="middlegame",
        eval_before_cp=-300 * sign, eval_after_cp=(-300 - loss) * sign,
        move_quality=MoveQuality.GOOD, event_type=MoveEventType.QUIET,
    )
    assert classify_move_event(event) == expected
