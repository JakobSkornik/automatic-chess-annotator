"""Per-move comment resolution: the opening/LLM/facts cascade, PV variations,
and the academic reasoning-trace debug block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
from chess.pgn import Game

from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.GameJson import Variation
from app.models.Move import Move

from .board_utils import _board_before_mainline_move, _score_to_move_score
from .constants import DEFAULT_ANALYSIS_DEPTH


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
        return (
            f"Continues the same theme as around ply {move_event.commentary_stub_ref_ply} "
            f"(see that move's commentary)."
        )
    return _facts_floor_comment(move_event) or _key_moment_comment(
        move_event, key_moment
    )


def _resolve_move_comment(
    row: AnalyzedMoveData,
    analyzed_move: Move,
    move_event: MoveEvent | None,
    key_moment: str | None,
    *,
    side_ok: bool,
) -> _MoveComment:
    """Resolve the comment cascade: opening -> LLM -> stub/facts/key-moment floor.

    Prose is reserved for key moments (an LLM pass ran for them); other
    out-of-book moves carry no prose — their structured facts panel stands
    alone. A non-selected commentary side stays fully silent.
    """
    mc = _MoveComment()
    if (row.phase_raw or "") == "early":
        mc.comment = _opening_comment(analyzed_move)
    elif move_event and move_event.key_moment_type and side_ok:
        _apply_llm_comment(mc, analyzed_move)
        if not mc.comment:
            mc.comment = _fallback_comment(move_event, key_moment)
    return mc


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
    }


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
