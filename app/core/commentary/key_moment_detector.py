from __future__ import annotations

import logging
import os

from app.models.Move import Move

logger = logging.getLogger(__name__)


def decisive_eval_cp() -> int:
    """Half-width of the "still a real game" interval, in centipawns (Guid).

    When both the played move and the engine's suggestion evaluate beyond this
    (default +-2.00, per Guid's 2006 World-Champions paper — the advisor's own
    published methodology, confirmed unchanged per his direct feedback: this
    exact +-2.00 gate is what correctly discards e.g. a played +4.80 vs. best
    +2.84 as "not a mistake"), the position is already decided: a player with
    a winning/lost game often plays a "good enough" or practical move rather
    than the engine's best, so imprecise/"missed opportunity" moves there must
    not be flagged as mistakes. Tunable via ``DECISIVE_EVAL_CP`` (the advisor
    mentioned +-3.00..+-5.00 as an option worth exploring, but gave no case
    that this narrower +-2.00 gate handles incorrectly, so it is left as-is
    here; see ``DECISIVE_CLAIM_CP`` in rules/constants.py for the separate,
    now-tightened positional-claim-suppression band).
    """
    try:
        return int(os.environ.get("DECISIVE_EVAL_CP", "200"))
    except ValueError:
        return 200


# ---------------------------------------------------------------------------
# Priority tiers – lower number = higher priority.
# When multiple triggers fire on the same move the highest-priority one wins.
# ---------------------------------------------------------------------------
KEY_MOMENT_PRIORITY: dict[str, int] = {
    "brilliant": 1,
    "blunder": 1,
    "critical_decision": 2,
    "structural_transformation": 2,
    "kingside_attack": 2,
    "great_move": 3,
    "mistake": 3,
    "king_safety_crisis": 3,
    "hidden_inflection": 4,
    "initiative_shift": 4,
    "piece_activation": 4,
    "good_defense": 4,
    "inaccuracy": 5,
    "missed_opportunity": 5,
    "opening_transition": 6,
    "endgame_transition": 6,
}


# --- Score-swing key-moment bands (centipawns, mover-POV) ---
BLUNDER_SWING_CP = -200
MISTAKE_SWING_CP = -100
INACCURACY_SWING_CP = -50
PRESSURE_EVAL_CP = 100  # |eval| under which a side is "under pressure" (good_defense)
MISSED_OPPORTUNITY_CP = 200  # best move this much better than the move played

# --- PV-comparison key-moment thresholds (centipawns) ---
GREAT_MOVE_GAP_CP = 120  # best move clearly better than the 2nd best
CRITICAL_TOP2_GAP_CP = 20  # top two moves this close = a real decision
CRITICAL_EVAL_SWING_CP = 60  # tangible swing required to call a move critical
CRITICAL_MATERIAL_SWING_CP = 100  # material story forks by this much
CRITICAL_KING_EXPOSURE_JUMP = 2  # king-exposure story forks by this much

# --- Strategic-trigger thresholds ---
KING_SAFETY_CRISIS_EXPOSURE_JUMP = 3
INITIATIVE_MOBILITY_DELTA = 8
INITIATIVE_ATTACKERS_DELTA = 2
ACTIVATION_CENTRALIZATION_GAIN = 1.0
ACTIVATION_PASSIVE_MAX = 1.5  # only from a passive (low-centralization) start


def _eval_swing_class(perspective_change: int, decided: bool) -> list[str]:
    """Blunder/mistake/inaccuracy from the mover-POV eval drop (live games only)."""
    if decided:
        return []
    if perspective_change <= BLUNDER_SWING_CP:
        return ["blunder"]
    if perspective_change <= MISTAKE_SWING_CP:
        return ["mistake"]
    if perspective_change <= INACCURACY_SWING_CP:
        return ["inaccuracy"]
    return []


def _good_defense(
    prev_score: int | None, is_white_move: bool, perspective_change: int
) -> list[str]:
    """Holding or improving the eval while under pressure (bad for the mover)."""
    if prev_score is None or perspective_change < 0:
        return []
    under_pressure = (
        prev_score <= -PRESSURE_EVAL_CP
        if is_white_move
        else prev_score >= PRESSURE_EVAL_CP
    )
    return ["good_defense"] if under_pressure else []


def _missed_opportunity(
    pvs_for_move: list[list[Move]] | None,
    current_move: Move,
    is_white_move: bool,
    decided: bool,
) -> list[str]:
    """The engine's best move was much better than the one played."""
    if decided or not (pvs_for_move and pvs_for_move[0]):
        return []
    best = pvs_for_move[0][0]
    if not best or best.score is None or current_move.score is None:
        return []
    diff = best.score - current_move.score
    if not is_white_move:
        diff = -diff
    return ["missed_opportunity"] if diff >= MISSED_OPPORTUNITY_CP else []


def _is_recapture(current_move: Move, previous_move: Move | None) -> bool:
    """The played move captures on the very square the opponent just moved to —
    an obvious forced recapture, not an instructive 'great' find (Guid)."""
    cur = getattr(current_move, "move", None)
    prev = getattr(previous_move, "move", None) if previous_move else None
    if not cur or not prev or len(cur) < 4 or len(prev) < 4:
        return False
    return cur[2:4] == prev[2:4]


def _great_move(
    pvs_for_move: list[list[Move]] | None,
    current_move: Move,
    is_white_move: bool,
    previous_move: Move | None = None,
) -> list[str]:
    """The played move is best and clearly better than the second choice.

    Obvious recaptures are excluded: a forced recapture often has a large gap to
    the (bad) alternative of not recapturing, but it is not a 'great' move."""
    if not pvs_for_move or len(pvs_for_move) < 2:
        return []
    pv1 = pvs_for_move[0][0] if pvs_for_move[0] else None
    pv2 = pvs_for_move[1][0] if pvs_for_move[1] else None
    if not pv1 or not pv2 or getattr(pv1, "move", None) != current_move.move:
        return []
    if pv1.score is None or pv2.score is None:
        return []
    if _is_recapture(current_move, previous_move):
        return []
    gap = pv1.score - pv2.score if is_white_move else pv2.score - pv1.score
    return ["great_move"] if gap >= GREAT_MOVE_GAP_CP else []


def _king_story_forks(prev_hf: dict, curr_hf: dict) -> bool:
    """A king-exposure swing big enough to make the choice critical."""
    for clr in ("white", "black"):
        pe = (prev_hf.get(clr) or {}).get("kingExposure")
        ce = (curr_hf.get(clr) or {}).get("kingExposure")
        if (
            isinstance(pe, (int, float))
            and isinstance(ce, (int, float))
            and abs(ce - pe) >= CRITICAL_KING_EXPOSURE_JUMP
        ):
            return True
    return False


def _material_story_forks(prev_hf: dict, curr_hf: dict) -> bool:
    """A material swing big enough to make the choice critical."""

    def total(hidden_features: dict):
        diff = (hidden_features.get("material") or {}).get("diff")
        return diff.get("total") if isinstance(diff, dict) else None

    pd, cd = total(prev_hf), total(curr_hf)
    return (
        isinstance(pd, (int, float))
        and isinstance(cd, (int, float))
        and abs(cd - pd) >= CRITICAL_MATERIAL_SWING_CP
    )


def _king_safety_crisis(curr_hf: dict, prev_hf: dict, side: str) -> list[str]:
    curr = (curr_hf.get(side) or {}).get("kingExposure")
    prev = (prev_hf.get(side) or {}).get("kingExposure")
    if curr is None or prev is None:
        return []
    return (
        ["king_safety_crisis"]
        if curr - prev >= KING_SAFETY_CRISIS_EXPOSURE_JUMP
        else []
    )


# An attack arc: the mover's king-zone attack count jumps by this many pieces
# in one move (a piece arriving in the enemy king's zone — sacrifice/lift/
# rook to the open file against the king).
KING_ZONE_ATTACK_JUMP = 2


def _zone_attacks_for(hf: dict, key: str) -> int | None:
    """A side's king-zone attack count from the Guid vector dump (if present)."""
    guid = hf.get("_guid") if isinstance(hf.get("_guid"), dict) else None
    if not guid:
        return None
    entry = guid.get(key)
    if isinstance(entry, dict) and entry.get("flag") is not None:
        return int(entry["flag"])
    return None


def _kingside_attack_arc(current_move: Move, previous_move: Move | None) -> list[str]:
    """A piece newly joins the attack on the enemy king's zone.

    The mover's own zone count is read from the Guid vector dump; a jump of
    ``KING_ZONE_ATTACK_JUMP`` or more in one move means a piece arrived in
    the enemy king's zone (sacrifice, lift, rook to the file against the king)."""
    is_white = current_move.depth % 2 == 1
    own_key = "WHITE_KING_ZONE_ATTACKS" if is_white else "BLACK_KING_ZONE_ATTACKS"
    curr = _zone_attacks_for(current_move.hiddenFeatures or {}, own_key)
    prev = (
        _zone_attacks_for(previous_move.hiddenFeatures or {}, own_key)
        if previous_move
        else None
    )
    if curr is None or prev is None:
        return []
    return ["kingside_attack"] if curr - prev >= KING_ZONE_ATTACK_JUMP else []


def _initiative_shift(curr_hf: dict, prev_hf: dict, side: str) -> list[str]:
    cm = (curr_hf.get(side) or {}).get("mobility")
    pm = (prev_hf.get(side) or {}).get("mobility")
    ca = (curr_hf.get(side) or {}).get("attackingPieces")
    pa = (prev_hf.get(side) or {}).get("attackingPieces")
    if any(v is None for v in (cm, pm, ca, pa)):
        return []
    gained_mobility = cm - pm >= INITIATIVE_MOBILITY_DELTA
    gained_attackers = ca - pa >= INITIATIVE_ATTACKERS_DELTA
    return ["initiative_shift"] if gained_mobility and gained_attackers else []


def _piece_activation(curr_hf: dict, prev_hf: dict, side: str) -> list[str]:
    curr = (curr_hf.get(side) or {}).get("centralization")
    prev = (prev_hf.get(side) or {}).get("centralization")
    if curr is None or prev is None:
        return []
    from_passive = prev <= ACTIVATION_PASSIVE_MAX
    big_gain = curr - prev >= ACTIVATION_CENTRALIZATION_GAIN
    return ["piece_activation"] if big_gain and from_passive else []


class KeyMomentDetector:
    """Detects key moments in a chess game.

    Returns the single highest-priority key moment type for a given move, or
    ``None`` if no trigger fires.
    """

    def __init__(self) -> None:
        self._prev_phase: str | None = None
        self._prev_pawn_structure_type: str | None = None
        self._last_book_depth: int = 0  # ply of last detected opening/book move
        self._fired_opening_transition: bool = False
        self._fired_endgame_transition: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        current_move: Move,
        previous_move: Move | None,
        pvs_for_move: list[list[Move]] | None,
        *,
        pv1_change_count: int = 0,
    ) -> str | None:
        """Return the highest-priority key moment type, or *None*."""
        candidates: list[str] = []

        # --- Score-based triggers (require both moves to have scores) ---
        if (
            previous_move
            and current_move.score is not None
            and previous_move.score is not None
        ):
            candidates.extend(
                self._score_based(
                    current_move,
                    previous_move,
                    pvs_for_move,
                    pv1_change_count=pv1_change_count,
                )
            )

        # --- Feature / strategic triggers ---
        candidates.extend(
            self._strategic_triggers(current_move, previous_move, pvs_for_move)
        )

        if not candidates:
            # Update tracking state even when no trigger fires
            self._update_tracking(current_move)
            return None

        # Pick the highest-priority (lowest number) candidate
        best = min(candidates, key=lambda c: KEY_MOMENT_PRIORITY.get(c, 99))
        logger.info(
            f"Move {current_move.depth}: key moment = {best} (candidates: {candidates})"
        )

        self._update_tracking(current_move)
        return best

    # ------------------------------------------------------------------
    # Score-based triggers
    # ------------------------------------------------------------------

    def _score_based(
        self,
        current_move: Move,
        previous_move: Move,
        pvs_for_move: list[list[Move]] | None,
        *,
        pv1_change_count: int = 0,
    ) -> list[str]:
        is_white_move = current_move.depth % 2 == 1
        score_change = current_move.score - previous_move.score  # type: ignore[operator]
        perspective_change = score_change if is_white_move else -score_change
        prev_hf = (previous_move.hiddenFeatures or {}) if previous_move else {}
        curr_hf = current_move.hiddenFeatures or {}
        self._log_score(current_move, previous_move, is_white_move, perspective_change)

        # Already-decided position: an imprecise move past the decisive interval
        # is not a real mistake (Guid) — the negative triggers self-suppress.
        decisive = decisive_eval_cp()
        decided = (
            abs(current_move.score) > decisive and abs(previous_move.score) > decisive
        )
        return [
            *_eval_swing_class(perspective_change, decided),
            *_good_defense(previous_move.score, is_white_move, perspective_change),
            *_missed_opportunity(pvs_for_move, current_move, is_white_move, decided),
            *_great_move(pvs_for_move, current_move, is_white_move, previous_move),
            *self._critical_decision(
                pvs_for_move, current_move, previous_move, prev_hf, curr_hf
            ),
        ]

    @staticmethod
    def _log_score(
        current_move: Move,
        previous_move: Move,
        is_white_move: bool,
        perspective_change: int,
    ) -> None:
        logger.info(
            "Move %s (%s): Prev=%s, Curr=%s, PerspChange=%s",
            current_move.depth,
            "White" if is_white_move else "Black",
            previous_move.score,
            current_move.score,
            perspective_change,
        )

    def _critical_decision(
        self,
        pvs_for_move: list[list[Move]] | None,
        current_move: Move,
        previous_move: Move,
        prev_hf: dict,
        curr_hf: dict,
    ) -> list[str]:
        """Top-two moves are close but the position forks (structure/king/material)."""
        if not pvs_for_move or len(pvs_for_move) < 2:
            return []
        pv1 = pvs_for_move[0][0] if pvs_for_move[0] else None
        pv2 = pvs_for_move[1][0] if pvs_for_move[1] else None
        if not pv1 or not pv2 or pv1.score is None or pv2.score is None:
            return []
        if abs(pv1.score - pv2.score) > CRITICAL_TOP2_GAP_CP:
            return []
        forks = (
            self._structure_forks(pv1, pv2)
            or _king_story_forks(prev_hf, curr_hf)
            or _material_story_forks(prev_hf, curr_hf)
        )
        eval_swing_abs = abs(current_move.score - previous_move.score)  # type: ignore[operator]
        if eval_swing_abs >= CRITICAL_EVAL_SWING_CP and forks:
            return ["critical_decision"]
        return []

    def _structure_forks(self, pv1: Move, pv2: Move) -> bool:
        ps1, ps2 = self._pawn_structure_type(pv1), self._pawn_structure_type(pv2)
        return bool(ps1 and ps2 and ps1 != ps2)

    def _strategic_triggers(
        self,
        current_move: Move,
        previous_move: Move | None,
        pvs_for_move: list[list[Move]] | None,
    ) -> list[str]:
        curr_hf = current_move.hiddenFeatures or {}
        prev_hf = (previous_move.hiddenFeatures if previous_move else None) or {}
        side = "white" if current_move.depth % 2 == 1 else "black"
        return [
            *self._structural_transformation(curr_hf),
            *_king_safety_crisis(curr_hf, prev_hf, side),
            *_kingside_attack_arc(current_move, previous_move),
            *_initiative_shift(curr_hf, prev_hf, side),
            *_piece_activation(curr_hf, prev_hf, side),
            *self._opening_transition(current_move),
            *self._endgame_transition(current_move),
        ]

    def _structural_transformation(self, curr_hf: dict) -> list[str]:
        curr_ps = (curr_hf.get("pawnStructure") or {}).get("centerType")
        changed = (
            self._prev_pawn_structure_type
            and curr_ps
            and curr_ps != self._prev_pawn_structure_type
        )
        return ["structural_transformation"] if changed else []

    def _opening_transition(self, current_move: Move) -> list[str]:
        left_book = (
            not self._fired_opening_transition
            and self._prev_phase in ("early", "opening")
            and current_move.phase
            and current_move.phase not in ("early", "opening")
        )
        if not left_book:
            return []
        self._fired_opening_transition = True
        self._last_book_depth = current_move.depth
        return ["opening_transition"]

    def _endgame_transition(self, current_move: Move) -> list[str]:
        entered_endgame = (
            self._prev_phase
            and current_move.phase
            and not self._fired_endgame_transition
            and self._prev_phase in ("opening", "mid", "middlegame", "early")
            and current_move.phase in ("end", "endgame")
        )
        if not entered_endgame:
            return []
        self._fired_endgame_transition = True
        return ["endgame_transition"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_tracking(self, move: Move) -> None:
        """Update internal tracking state after processing a move."""
        self._prev_phase = move.phase
        hidden_features = move.hiddenFeatures or {}
        ps = hidden_features.get("pawnStructure")
        if isinstance(ps, dict):
            self._prev_pawn_structure_type = ps.get("centerType")

    @staticmethod
    def _pawn_structure_type(move: Move) -> str | None:
        """Extract pawn structure center type from a move's hiddenFeatures."""
        hidden_features = move.hiddenFeatures
        if not isinstance(hidden_features, dict):
            return None
        ps = hidden_features.get("pawnStructure")
        if not isinstance(ps, dict):
            return None
        return ps.get("centerType")
