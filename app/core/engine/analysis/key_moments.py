"""Key-moment post-processing: back-to-back suppression, the post-facts
brilliant/best-move promotion, and the NAG-style move-symbol mapping."""

from __future__ import annotations

from typing import Any

import chess

from app.core.commentary.features.envisioned import is_quiescent
from app.core.commentary.features.tactical_motifs.sacrifice_core import (
    persistent_deficit,
)
from app.models.chess_events import GameAnalysisContext, MoveEvent
from .constants import MATE_SCORE


def _comment_selection(move_event: MoveEvent, facts: Any = None) -> dict[str, Any]:
    """Final substantive-comment gate; capture evidence alone is no exception.

    Guid §5.3 requires quiet starting and envisioned positions. Allowing
    verified errors in noisy positions is application policy,
    rather than numerical rules prescribed by the dissertation.
    """
    facts = facts if facts is not None else move_event.comment_facts
    board = chess.Board(move_event.fen_before)
    after = chess.Board(move_event.fen_after)
    move = chess.Move.from_uci(move_event.uci)
    replay = board.copy(stack=False)
    if move not in replay.legal_moves:
        return {"allowed": False, "reason": "illegal_played_move",
                "classification": move_event.key_moment_type}
    replay.push(move)
    if replay.fen() != after.fen():
        return {"allowed": False, "reason": "played_position_mismatch",
                "classification": move_event.key_moment_type}
    line = getattr(facts, "display_line", None)
    start_quiet = is_quiescent(board)
    leaf_quiet = bool(line and is_quiescent(chess.Board(line.leaf_fen)))
    forcing = board.is_capture(move) or bool(move.promotion) or after.is_check()
    quality = getattr(move_event.move_quality, "value", move_event.move_quality)
    sign = 1 if board.turn == chess.WHITE else -1
    best_eval = move_event.best_move_eval_cp
    played_eval = move_event.eval_after_cp
    best_finite = best_eval is not None and abs(best_eval) <= MATE_SCORE - 1000
    played_finite = played_eval is not None and abs(played_eval) <= MATE_SCORE - 1000
    loss = (sign * (best_eval - played_eval)
            if best_finite and played_finite else None)
    try:
        different_best = bool(move_event.best_move_uci != move_event.uci and
                              chess.Move.from_uci(move_event.best_move_uci or "") in board.legal_moves)
    except ValueError:
        different_best = False
    alternative = getattr(facts, "better_alternative", None)
    played_mate = getattr(facts, "eval_mate", None)
    best_mate = (getattr(alternative, "eval_mate", None)
                 if alternative and not alternative.is_inferior and
                 getattr(alternative, "uci", None) == move_event.best_move_uci else None)
    if played_mate is None and played_eval is not None and not played_finite:
        played_mate = (1 if played_eval > 0 else -1) * (MATE_SCORE - abs(played_eval))
    if best_mate is None and best_eval is not None and not best_finite:
        best_mate = (1 if best_eval > 0 else -1) * (MATE_SCORE - abs(best_eval))
    # Forced mate outranks finite centipawns; shorter winning mates and longer
    # losing mates alone do not establish a missed opportunity.
    mate_loss = bool(different_best and (
        (best_mate is not None and sign * best_mate > 0 and
         (played_finite or (played_mate is not None and sign * played_mate < 0))) or
        (played_mate is not None and sign * played_mate < 0 and
         (best_finite or (best_mate is not None and sign * best_mate > 0)))))
    error = quality in {"mistake", "blunder"} and different_best and (mate_loss or (loss is not None and loss > 100))
    missed = (move_event.key_moment_type == "missed_opportunity" and different_best
              and (mate_loss or (loss is not None and loss >= 200)))
    if after.is_game_over() or getattr(facts, "terminal_result", None):
        reason = "terminal_result"
    elif error:
        reason = "verified_quality_error"
    elif missed:
        reason = "verified_missed_opportunity"
    elif not start_quiet:
        reason = "nonquiescent_start"
    elif not leaf_quiet:
        reason = "nonquiescent_or_missing_endpoint"
    elif not getattr(line, "evidence_valid", False):
        reason = "invalid_endpoint_evidence"
    else:
        reason = "quiescent_move_and_endpoint"
    return {"allowed": reason in {"terminal_result", "verified_quality_error",
                                  "verified_missed_opportunity", "quiescent_move_and_endpoint"},
            "reason": reason, "start_quiescent": start_quiet,
            "leaf_quiescent": leaf_quiet, "forcing_played_move": forcing,
            "verified_loss_cp": loss, "verified_mate_loss": mate_loss,
            "classification": move_event.key_moment_type,
            "quiescence_method": "static_check_and_en_prise_heuristic"}


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
        if key_moment in ("blunder", "mistake", "critical_decision", "checkmate", "game_end"):
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


# Brilliant-sacrifice detection (post-facts). A real sacrifice keeps the mover
# DOWN this much material at the END of the played line — material regained
# within the sequence (a recapture/clearance such as knight-for-bishop) is NOT
# a sacrifice (Guid). It must also leave the mover clearly winning, so the
# sacrifice demonstrably works; a move that merely holds equality is not "!!".
BRILLIANT_SAC_LEAF_CP = 150
BRILLIANT_MIN_EVAL_CP = 100


def _played_is_best(move_event: MoveEvent) -> bool:
    if move_event.best_move_uci and move_event.uci:
        return move_event.best_move_uci == move_event.uci
    return bool(move_event.move_quality and move_event.move_quality.value == "best")


def _is_sacrifice(facts: Any) -> bool:
    """Material is still down for the mover at the END of the line. A transient
    dip that is regained within the sequence is not a sacrifice.

    Uses the shared ``persistent_deficit`` core (sacrifice_core.py) also used
    by the general-purpose ``TacticalMotif.SACRIFICE`` detector — this is the
    brilliancy-specific caller, layering ``_played_is_best`` and
    ``_eval_clearly_winning`` on top in ``_promote_key_moment`` below.
    """
    dl = facts.display_line
    series = (dl.feature_series.get("MATERIAL_BALANCE") if dl else None) or []
    if len(series) < 2:
        return False
    sign = 1 if facts.mover == "White" else -1
    return persistent_deficit(series, sign) <= -BRILLIANT_SAC_LEAF_CP


def _eval_clearly_winning(facts: Any) -> bool:
    """The mover is clearly better after the move — i.e. the sacrifice works.
    Merely holding equality does not make a move brilliant."""
    if facts.eval_cp is None:
        return facts.eval_mate is not None and (
            (facts.eval_mate > 0) == (facts.mover == "White")
        )
    sign = 1 if facts.mover == "White" else -1
    return sign * facts.eval_cp >= BRILLIANT_MIN_EVAL_CP


def _promote_key_moment(
    move_event: MoveEvent, facts: Any, kept_claims: list
) -> str | None:
    """Upgrade the key-moment type using the now-available CommentFacts.

    - ``brilliant``: a best move that gives up material lasting to the line's
      end yet stays clearly winning (highest priority).
    - ``best_move``: an instructive top move (>= 1 fired claim) that nothing
      else flagged. Decisive positions self-exclude (their claims are emptied).
    """
    if getattr(facts, "terminal_result", None):
        return move_event.key_moment_type or "game_end"
    if _played_is_best(move_event):
        if _is_sacrifice(facts) and _eval_clearly_winning(facts):
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
