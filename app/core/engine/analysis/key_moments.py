"""Key-moment post-processing: back-to-back suppression, the post-facts
brilliant/best-move promotion, and the NAG-style move-symbol mapping."""

from __future__ import annotations

from typing import Any

from app.models.chess_events import GameAnalysisContext, MoveEvent


def _apply_back_to_back_key_moment_suppression(
    move_events: list[MoveEvent],
    context: GameAnalysisContext,
) -> None:
    """Second of two same-type key moments within 2 plies loses LLM pass; gets stub reference."""
    last_ply: int | None = None
    last_type: str | None = None
    for mi, move_event in enumerate(move_events):
        key_moment = move_event.key_moment_type
        if not key_moment:
            continue
        if key_moment in ("blunder", "mistake", "critical_decision"):
            last_ply, last_type = move_event.ply, key_moment
            continue
        if (
            last_type == key_moment
            and last_ply is not None
            and (move_event.ply - last_ply) <= 2
        ):
            updated = move_event.model_copy(
                update={
                    "key_moment_type": None,
                    "brief_commentary": True,
                    "commentary_stub_ref_ply": last_ply,
                    "is_critical": True,
                }
            )
            move_events[mi] = updated
            context.move_events[mi] = updated
        else:
            last_ply, last_type = move_event.ply, key_moment


# Brilliant-sacrifice detection (post-facts): along the played PV the mover
# gives up at least this much material yet keeps an equal-or-better eval.
BRILLIANT_SAC_TROUGH_CP = 150


def _played_is_best(move_event: MoveEvent) -> bool:
    if move_event.best_move_uci and move_event.uci:
        return move_event.best_move_uci == move_event.uci
    return bool(move_event.move_quality and move_event.move_quality.value == "best")


def _is_sacrifice(facts: Any) -> bool:
    """The mover's material dips >= threshold somewhere along the played line."""
    dl = facts.display_line
    series = (dl.feature_series.get("MATERIAL_BALANCE") if dl else None) or []
    if len(series) < 2:
        return False
    sign = 1 if facts.mover == "White" else -1
    start = series[0]
    trough = min(sign * (v - start) for v in series)
    return trough <= -BRILLIANT_SAC_TROUGH_CP


def _eval_holds_or_improves(facts: Any) -> bool:
    """Mover is at least equal after the move and no worse than before it."""
    if facts.eval_cp is None:
        return facts.eval_mate is not None and (
            (facts.eval_mate > 0) == (facts.mover == "White")
        )
    sign = 1 if facts.mover == "White" else -1
    after = sign * facts.eval_cp
    if after < 0:  # mover ends up worse — not a sound sacrifice
        return False
    if facts.eval_before_cp is None:
        return True
    return after >= sign * facts.eval_before_cp - 20


def _promote_key_moment(
    move_event: MoveEvent, facts: Any, kept_claims: list
) -> str | None:
    """Upgrade the key-moment type using the now-available CommentFacts.

    - ``brilliant``: a best-move sacrifice that holds/improves the eval (highest
      priority — overrides whatever the detector flagged).
    - ``best_move``: an instructive top move (>= 1 fired claim) that nothing
      else flagged. Decisive positions self-exclude (their claims are emptied).
    """
    if _played_is_best(move_event):
        if _is_sacrifice(facts) and _eval_holds_or_improves(facts):
            return "brilliant"
        if move_event.key_moment_type is None and kept_claims:
            return "best_move"
    return move_event.key_moment_type


# Key-moment classification -> NAG-style move symbol shown in the move list.
_ANNOTATION_BY_KEY_MOMENT: dict[str, str] = {
    "brilliant": "!!",
    "best_move": "!",
    "great_move": "!",
    "inaccuracy": "?!",
    "missed_opportunity": "?!",
    "mistake": "?",
    "blunder": "??",
}


def _annotation_symbol(key_moment_type: str | None) -> str | None:
    return _ANNOTATION_BY_KEY_MOMENT.get(key_moment_type or "")
