"""Per-move comment resolution: the opening/LLM/facts cascade, PV variations,
and the academic reasoning-trace debug block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
from chess.pgn import Game

from app.core.commentary.rules.constants import INFERIOR_ALT_WEAKER_CP
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.GameJson import Variation
from app.models.Move import Move

from .board_utils import _board_before_mainline_move, _score_to_move_score
from .constants import DEFAULT_ANALYSIS_DEPTH
from app.core.commentary.rendering_contracts import WHOLE_ANNOTATION_CONTRACTS


@dataclass
class _MoveComment:
    """A move's resolved comment."""

    comment: str | None = None


def _opening_comment(analyzed_move: Move) -> str | None:
    hidden = analyzed_move.hiddenFeatures or {}
    opening = hidden.get("_opening") if isinstance(hidden, dict) else None
    if isinstance(opening, dict) and opening.get("comment"):
        return str(opening["comment"])
    return None


def _apply_llm_comment(mc: _MoveComment, analyzed_move: Move) -> None:
    """Overlay the LLM-authored comment, per-level texts, and motifs onto ``mc``."""
    hidden = analyzed_move.hiddenFeatures or {}
    llm = hidden.get("_llm") if isinstance(hidden, dict) else None
    if not isinstance(llm, dict):
        return
    if llm.get("comment"):
        mc.comment = str(llm["comment"])


def _facts_floor_comment(move_event: MoveEvent | None) -> str | None:
    """Deterministic Guid template (verdict + line + eval) for a move with facts."""
    if not move_event or move_event.comment_facts is None:
        return None
    try:
        from app.core.commentary.phases.composer import render_facts_template

        return render_facts_template(move_event.comment_facts)
    except Exception:
        return None


def _key_moment_comment(
    move_event: MoveEvent | None, key_moment: str | None
) -> str | None:
    if not (key_moment and move_event):
        return None
    label = key_moment.replace("_", " ").capitalize()
    swing = move_event.eval_swing_cp
    if swing is None:
        return label
    return f"{label} (Eval swing: {swing / 100:+.2f} pawns, White POV step)"


def _fallback_comment(
    move_event: MoveEvent | None, key_moment: str | None
) -> str | None:
    """Comment of last resort: brief stub, else facts floor, else key-moment line."""
    move_event = move_event
    if (
        move_event
        and move_event.brief_commentary
        and move_event.commentary_stub_ref_ply is not None
    ):
        ref_ply = move_event.commentary_stub_ref_ply
        ref_move = (ref_ply + 1) // 2
        ref_notation = f"{ref_move}." if ref_ply % 2 else f"{ref_move}..."
        return (
            f"Continues the same theme as around move {ref_notation} "
            f"(see that move's commentary)."
        )
    return _facts_floor_comment(move_event) or _key_moment_comment(
        move_event, key_moment
    )


# A "!"/"!!" annotation can stand on its own — these classifications mark a good
# move without obliging a comment. (Negative ones always have a consequence to
# explain, so they are never silenced.)
_OPTIONAL_ANNOTATIONS = frozenset({"best_move", "great_move", "brilliant"})


def _nothing_instructive_to_say(move_event: MoveEvent) -> bool:
    """True when an optional "!"-type move has no instructive content. The
    annotation then stands alone rather than emitting a hollow "Best move" line
    (Guid: if there is nothing smart to say, say nothing).

    There is something to say when either (a) a rule fired a claim, or (b) the
    second-best move is *clearly* weaker, so the contrast itself is instructive.
    A merely-comparable runner-up adds nothing."""
    if move_event.key_moment_type not in _OPTIONAL_ANNOTATIONS:
        return False
    facts = move_event.comment_facts
    if facts is None:
        return True
    if facts.claims:
        return False
    alt = facts.better_alternative
    if alt is not None and alt.is_inferior:
        gap = abs((alt.eval_cp or 0) - (facts.eval_cp or 0))
        return gap < INFERIOR_ALT_WEAKER_CP
    return alt is None


def _resolve_move_comment(
    row: AnalyzedMoveData,
    analyzed_move: Move,
    move_event: MoveEvent | None,
    key_moment: str | None,
    *,
    side_ok: bool,
    seen_transitions: dict[str, int] | None = None,
) -> _MoveComment:
    """Resolve the comment cascade: opening -> LLM -> stub/facts/key-moment floor,
    plus a phase-transition sentence.

    Prose is reserved for key moments (an LLM pass ran for them); other
    out-of-book moves carry no prose — their structured facts panel stands
    alone. A non-selected commentary side stays fully silent.

    A move that back-to-back suppression demoted (``brief_commentary``) has no
    ``key_moment_type`` of its own anymore — the LLM pass was deliberately
    skipped for it — but it still owes the reader the short stub reference the
    suppression promised, so it gets its own branch below rather than falling
    through to silence.
    """
    mc = _MoveComment()
    if move_event and move_event.comment_facts and move_event.comment_facts.terminal_result:
        mc.comment = _facts_floor_comment(move_event) if side_ok else None
        return mc
    if (row.phase_raw or "") == "early":
        mc.comment = _opening_comment(analyzed_move)
    elif move_event and move_event.key_moment_type and side_ok:
        _apply_llm_comment(mc, analyzed_move)
        # Say nothing rather than emit a hollow floor line for an optional "!"
        # move that carries no instructive facts (Guid). The LLM comment, if it
        # found something to say, is still honored above.
        if not mc.comment and not _nothing_instructive_to_say(move_event):
            mc.comment = _fallback_comment(move_event, key_moment)
    elif move_event and move_event.brief_commentary and side_ok:
        mc.comment = _fallback_comment(move_event, key_moment)
    llm = (analyzed_move.hiddenFeatures or {}).get("_llm", {})
    whole_annotation = (isinstance(llm, dict) and llm.get("rendering_contract") in WHOLE_ANNOTATION_CONTRACTS
                        and llm.get("semantic_review") in {"accepted", "disabled"})
    if not whole_annotation:
        _append_transition_sentence(mc, move_event, side_ok, seen_transitions)
    return mc


# A queen trade usually drops the piece count through the endgame threshold on
# the very next ply, and both boundaries say "we are in the endgame now". Only
# the first is worth a sentence; a later boundary this close is a restatement.
TRANSITION_QUIET_PLIES = 8


def _append_transition_sentence(
    mc: _MoveComment,
    move_event: MoveEvent | None,
    side_ok: bool,
    seen_transitions: dict[str, int] | None,
) -> None:
    """Add the phase-boundary sentence, alongside any prose already resolved.

    The move that takes the queens off is usually also a capture worth its own
    comment, so gating the sentence on an empty comment lost it precisely on the
    moves where the transition matters most.
    """
    if not side_ok or move_event is None:
        return
    kind = move_event.phase_transition
    if not kind:
        return
    from app.core.commentary.features.envisioned import is_quiescent
    board = chess.Board(move_event.fen_after)
    if (board.is_game_over() or not is_quiescent(board)
            or (move_event.eval_after_cp is not None and abs(move_event.eval_after_cp) >= 999000)):
        return
    if seen_transitions is not None:
        if kind in seen_transitions:
            return
        if any(
            move_event.ply - ply <= TRANSITION_QUIET_PLIES
            for ply in seen_transitions.values()
        ):
            return
        seen_transitions[kind] = move_event.ply
    sentence = _transition_sentence(move_event, kind)
    mc.comment = f"{mc.comment.rstrip()} {sentence}" if mc.comment else sentence


def _transition_sentence(move_event: MoveEvent, kind: str) -> str:
    base = (
        "Both queens are off the board."
        if kind == "queens_off"
        else "The reduced material marks the transition to an endgame."
    )
    eval_after = move_event.eval_after_cp
    if eval_after is None:
        return base
    if abs(eval_after) > 150:
        pov = "White" if eval_after >= 0 else "Black"
        return f"{base} {pov} remains firmly better going into it."
    swing = move_event.eval_swing_cp
    if swing is not None and abs(swing) >= 80:
        return f"{base} The balance shifted here ({swing / 100:+.2f} for White)."
    return f"{base} The position stays balanced."


def _trace_pv_line(
    board_start: chess.Board, pv_sequence: list
) -> tuple[list[str], list[str]]:
    """SAN and post-move FEN for each ply of a PV, walked from ``board_start``."""
    san_line: list[str] = []
    fen_line: list[str] = []
    board = board_start.copy()
    for pm in pv_sequence:
        try:
            mv = chess.Move.from_uci(pm.move)
            san_line.append(board.san(mv))
            board.push(mv)
            fen_line.append(board.fen())
        except Exception:
            san_line.append(str(pm.move))
            fen_line.append("")
    return san_line, fen_line


def _build_variations(game: Game, move_index: int, pvs: list) -> list[Variation]:
    """Engine PVs -> Variation list (SAN + per-ply FENs) from the pre-move board."""
    board_start = _board_before_mainline_move(game, move_index)
    variations: list[Variation] = []
    for rank, pv_sequence in enumerate(pvs):
        if not pv_sequence:
            continue
        san_line, fen_line = _trace_pv_line(board_start, pv_sequence)
        variations.append(
            Variation(
                rank=rank + 1,
                move_san=san_line[0] if san_line else "",
                score=_score_to_move_score(pv_sequence[0].score),
                line=san_line,
                fens=fen_line,
                depth=DEFAULT_ANALYSIS_DEPTH,
            )
        )
    return variations


def _llm_rendering_debug(analyzed_move: Move) -> tuple[Any, Any]:
    """(facts_renderings, facts_contract_ok) the LLM stored under hiddenFeatures._llm."""
    hidden = analyzed_move.hiddenFeatures or {}
    llm = hidden.get("_llm") if isinstance(hidden, dict) else None
    if not isinstance(llm, dict):
        return None, None
    return llm.get("facts_renderings"), llm.get("facts_contract_ok")


def _facts_debug(
    facts: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    """(envisioned-line stats, fired-rule claims, muted claims) from CommentFacts."""
    if facts is None:
        return None, [], []
    envisioned_stats = None
    dl = facts.display_line
    if dl is not None:
        envisioned_stats = {
            "kept_plies": len(dl.line_san),
            "trimmed_plies": dl.trimmed_plies,
            "start_quiescent": dl.start_quiescent,
            "leaf_quiescent": dl.leaf_quiescent,
        }
    fired_rules = [
        {
            "rule_id": c.rule_id,
            "text": c.text,
            "delta_cp": c.delta_cp,
            "features": list(c.features_involved),
            "flag_note": c.flag_note,
        }
        for c in facts.claims
    ]
    return envisioned_stats, fired_rules, list(facts.muted_claims)


def _build_move_debug(
    analyzed_move: Move,
    move_event: MoveEvent | None,
    facts: Any,
    comment: str | None,
    phase_raw: str,
) -> dict[str, Any] | None:
    """Academic reasoning trace for a commented mid/endgame move (None otherwise)."""
    move_event = move_event
    if not (comment and move_event and phase_raw != "early"):
        return None
    renderings, contract_ok = _llm_rendering_debug(analyzed_move)
    envisioned_stats, fired_rules, muted = _facts_debug(facts)
    return {
        "eval_before_cp": move_event.eval_before_cp,
        "eval_after_cp": move_event.eval_after_cp,
        "eval_swing_cp": move_event.eval_swing_cp,
        "best_move_san": move_event.best_move_san,
        "best_move_eval_cp": move_event.best_move_eval_cp,
        "key_moment_type": move_event.key_moment_type,
        "move_quality": move_event.move_quality.value
        if move_event.move_quality
        else None,
        "envisioned": envisioned_stats,
        "fired_rules": fired_rules,
        "muted_claims": muted,
        "renderings": renderings,
        "contract_ok": contract_ok,
        "rendering_contract": ((analyzed_move.hiddenFeatures or {}).get("_llm") or {}).get("rendering_contract"),
        "commentary_validation": {
            key: ((analyzed_move.hiddenFeatures or {}).get("_llm") or {}).get(key)
            for key in ("prompt_version", "semantic_review", "review_scope", "selected_claims", "raw_proposal",
                        "review_decisions", "fallback_reason", "commentary_status", "missed_improvement", "claim_catalog")
        },
        "selection_metadata": getattr(facts, "selection_metadata", {}),
        # Plausible moves that only deep search exposed as bad (Guid Search
        # Module). Empty list = nothing found / scan disabled.
        "refutations": list(move_event.refutations or []),
        "phase_transition": move_event.phase_transition,
    }


def _comment_status(analyzed_move, move_event, comment, phase_raw, *, eligible, llm_done):
    """Expose provenance separately from selection and the visible template floor."""
    llm = ((analyzed_move.hiddenFeatures or {}).get("_llm") or {})
    status = llm.get("commentary_status") if comment else None
    reason = llm.get("commentary_reason") or llm.get("fallback_reason")
    terminal = bool(move_event and move_event.comment_facts and move_event.comment_facts.terminal_result)
    if phase_raw == "early" and comment:
        kind, status = "book", "book"
    elif move_event and move_event.brief_commentary and comment:
        kind, status = "reference", "reference"
    elif status:
        kind = "llm" if status.startswith("llm_") else "template"
    elif (eligible and not llm_done and not terminal and not llm.get("comment")
          and not (move_event and _nothing_instructive_to_say(move_event))):
        kind, status = "pending", "pending"
    elif comment:
        kind, status = "template", "template"
    else:
        kind, status = "none", "skipped"
        reason = "not_selected" if not eligible else "no_instructive_comment"
    return dict(has_comment=bool(comment and comment.strip()), comment_kind=kind,
                commentary_status=status, commentary_reason=reason)


def _resolve_comment_tokens(
    comment: str | None,
    move_event: MoveEvent | None,
) -> list[dict[str, Any]]:
    """Resolve interactive tokens for the comment."""
    if not (comment and move_event):
        return []
    try:
        from app.core.commentary.annotation_tokens import resolve_tokens_for_comment

        return resolve_tokens_for_comment(
            comment, move_event.fen_before, move_event.fen_after
        )
    except Exception:
        return []
