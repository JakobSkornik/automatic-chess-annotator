"""Classify a move into a coarse coach category (prompt selector)."""

from __future__ import annotations

import chess

from app.models.chess_events import (
    MoveCategory,
    MoveEvent,
    MoveQuality,
    StrategicMotif,
)

# Move-category thresholds.
UNDER_PRESSURE_CP = 100  # |eval| at/below which the mover is defending
DEFENSIVE_HOLD_TOLERANCE_CP = 40
BOOK_PLY_MAX = 22  # an in-theory best move counts as book up to this ply
BOOK_NEAR_PLY_MAX = 16
BOOK_NEAR_LOSS_CP = 35


def classify_move_event(move_event: MoveEvent) -> MoveCategory:
    """
    Deterministic category from motifs, eval, opening, and PV hints.
    Order follows mentor plan: tactical / quality / defensive / book / prophylactic / forcing / positional.
    """
    if move_event.tactical_motifs:
        return MoveCategory.TACTICAL

    if (
        move_event.move_quality == MoveQuality.BLUNDER
        or move_event.move_quality == MoveQuality.MISTAKE
    ):
        return MoveCategory.CRITICAL
    if move_event.move_quality == MoveQuality.INACCURACY:
        return MoveCategory.INACCURACY

    if (
        move_event.eval_before_cp is not None
        and move_event.eval_before_cp <= -UNDER_PRESSURE_CP
    ):
        if (
            move_event.eval_after_cp is not None
            and move_event.eval_after_cp
            >= move_event.eval_before_cp - DEFENSIVE_HOLD_TOLERANCE_CP
        ):
            return MoveCategory.DEFENSIVE

    if (
        move_event.phase == "opening"
        and move_event.opening_eco
        and move_event.ply <= BOOK_PLY_MAX
        and move_event.best_move_uci
        and move_event.uci == move_event.best_move_uci
    ):
        return MoveCategory.BOOK
    if (
        move_event.phase == "opening"
        and move_event.opening_eco
        and move_event.ply <= BOOK_NEAR_PLY_MAX
    ):
        loss = 0
        if (
            move_event.eval_after_cp is not None
            and move_event.best_move_eval_cp is not None
        ):
            loss = abs(move_event.best_move_eval_cp - move_event.eval_after_cp)
        if loss <= BOOK_NEAR_LOSS_CP:
            return MoveCategory.BOOK

    if StrategicMotif.PROPHYLAXIS in move_event.strategic_motifs:
        return MoveCategory.PROPHYLACTIC

    try:
        b = chess.Board(move_event.fen_before)
        m = chess.Move.from_uci(move_event.uci)
        if b.is_capture(m) or b.gives_check(m):
            return MoveCategory.FORCING
    except Exception:
        pass

    return MoveCategory.POSITIONAL
