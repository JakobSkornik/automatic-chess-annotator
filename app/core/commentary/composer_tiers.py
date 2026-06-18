"""Per-move composer budgeting: effort/token tiers, prose detail level, and the
RAG Top-K. Pure configuration + selection logic shared by the comment service
and the move pipeline.
"""

from __future__ import annotations

import os
from typing import Any

from app.models.chess_events import MoveEvent, MoveQuality

# ---------------------------------------------------------------------------
# Effort / token tiers — map key-moment types to pipeline depth & effort
# ---------------------------------------------------------------------------
TIER_FULL_HIGH = {"steps": 1, "effort": "medium", "max_tokens": 900}
TIER_FULL_LOW = {"steps": 1, "effort": "low", "max_tokens": 700}
TIER_SINGLE = {"steps": 1, "effort": "low", "max_tokens": 500}

KEY_MOMENT_TIERS: dict[str, dict[str, Any]] = {
    "brilliant": TIER_FULL_HIGH,
    "blunder": TIER_FULL_HIGH,
    "critical_decision": TIER_FULL_HIGH,
    "mistake": TIER_FULL_LOW,
    "structural_transformation": TIER_FULL_LOW,
    "king_safety_crisis": TIER_FULL_LOW,
    "hidden_inflection": TIER_SINGLE,
    "great_move": TIER_SINGLE,
    "inaccuracy": TIER_SINGLE,
    "good_defense": TIER_SINGLE,
    "initiative_shift": TIER_SINGLE,
    "piece_activation": TIER_SINGLE,
    "missed_opportunity": TIER_SINGLE,
    "opening_transition": TIER_SINGLE,
    "endgame_transition": TIER_SINGLE,
}

# ---------------------------------------------------------------------------
# Prose detail level
# ---------------------------------------------------------------------------
_FULL_DETAIL_MOMENTS = ("brilliant", "blunder", "critical_decision")
_COMPACT_DETAIL_MOMENTS = ("mistake", "structural_transformation", "king_safety_crisis")
OPENING_BOOK_MAX_MOVE_INDEX = 14
BOOK_QUIET_SWING_MAX_CP = 15


def detail_level_for_key_moment(move_event: MoveEvent) -> str:
    """Pick the prose detail level (full / compact / book / minimal) for a move."""
    km = move_event.key_moment_type or ""
    if km in _FULL_DETAIL_MOMENTS:
        return "full"
    if km in _COMPACT_DETAIL_MOMENTS:
        return "compact"
    if _is_book_opening_move(move_event) or _is_quiet_best_move(move_event):
        return "book"
    return "minimal"


def _is_book_opening_move(move_event: MoveEvent) -> bool:
    return (
        (move_event.phase or "").strip().lower() == "opening"
        and move_event.move_index <= OPENING_BOOK_MAX_MOVE_INDEX
        and move_event.move_quality in (MoveQuality.BEST, MoveQuality.EXCELLENT)
        and not move_event.key_moment_type
    )


def _is_quiet_best_move(move_event: MoveEvent) -> bool:
    return (
        not move_event.key_moment_type
        and bool(move_event.best_move_uci)
        and move_event.uci == move_event.best_move_uci
        and abs(move_event.eval_swing_cp or 0) < BOOK_QUIET_SWING_MAX_CP
    )


# ---------------------------------------------------------------------------
# RAG Top-K
# ---------------------------------------------------------------------------
TOP_K_DISABLED = 0
RAG_TOP_K_MIN = 1
RAG_TOP_K_MAX = 8
_TOP_K_BY_DETAIL = {"full": 3, "compact": 2, "minimal": 1}
_TOP_K_OPENING_ENDGAME = 3
_TOP_K_OPENING_ENDGAME_MINIMAL = 2
_PHASE_TOP_K_ENV_KEY = {
    "end": "RAG_TOP_K_ENDGAME",
    "endgame": "RAG_TOP_K_ENDGAME",
    "opening": "RAG_TOP_K_OPENING",
}
_OPENING_ENDGAME_PHASES = ("opening", "end", "endgame")


def compute_rag_top_k(detail: str, phase: str | None) -> int:
    """Per-phase Top-K default, overridden by RAG_TOP_K_<PHASE> or global RAG_TOP_K."""
    if detail == "book":
        return TOP_K_DISABLED
    phase_norm = (phase or "middlegame").strip().lower()
    override = _rag_top_k_override(phase_norm)
    if override is not None:
        return override
    if phase_norm in _OPENING_ENDGAME_PHASES:
        return (
            _TOP_K_OPENING_ENDGAME_MINIMAL
            if detail == "minimal"
            else _TOP_K_OPENING_ENDGAME
        )
    return _TOP_K_BY_DETAIL.get(detail, RAG_TOP_K_MIN)


def _rag_top_k_override(phase_norm: str) -> int | None:
    """Clamped Top-K from the environment, or None if unset/invalid."""
    phase_key = _PHASE_TOP_K_ENV_KEY.get(phase_norm, "RAG_TOP_K_MIDDLEGAME")
    for key in (phase_key, "RAG_TOP_K"):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return max(RAG_TOP_K_MIN, min(RAG_TOP_K_MAX, int(raw)))
            except ValueError:
                break
    return None
