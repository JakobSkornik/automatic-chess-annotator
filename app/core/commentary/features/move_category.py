"""Classify a move into a coarse coach category (prompt selector)."""

from __future__ import annotations

import chess

from app.models.chess_events import (
    MoveCategory,
    MoveEvent,
    MoveQuality,
    StrategicMotif,
)


def classify_move_event(me: MoveEvent) -> MoveCategory:
    """
    Deterministic category from motifs, eval, opening, and PV hints.
    Order follows mentor plan: tactical / quality / defensive / book / prophylactic / forcing / positional.
    """
    if me.tactical_motifs:
        return MoveCategory.TACTICAL

    if me.move_quality == MoveQuality.BLUNDER or me.move_quality == MoveQuality.MISTAKE:
        return MoveCategory.CRITICAL
    if me.move_quality == MoveQuality.INACCURACY:
        return MoveCategory.INACCURACY

    if me.eval_before_cp is not None and me.eval_before_cp <= -100:
        if me.eval_after_cp is not None and me.eval_after_cp >= me.eval_before_cp - 40:
            return MoveCategory.DEFENSIVE

    if (
        me.phase == "opening"
        and me.opening_eco
        and me.ply <= 22
        and me.best_move_uci
        and me.uci == me.best_move_uci
    ):
        return MoveCategory.BOOK
    if me.phase == "opening" and me.opening_eco and me.ply <= 16:
        loss = 0
        if me.eval_after_cp is not None and me.best_move_eval_cp is not None:
            loss = abs(me.best_move_eval_cp - me.eval_after_cp)
        if loss <= 35:
            return MoveCategory.BOOK

    if StrategicMotif.PROPHYLAXIS in me.strategic_motifs:
        return MoveCategory.PROPHYLACTIC

    try:
        b = chess.Board(me.fen_before)
        m = chess.Move.from_uci(me.uci)
        if b.is_capture(m) or b.gives_check(m):
            return MoveCategory.FORCING
    except Exception:
        pass

    return MoveCategory.POSITIONAL
