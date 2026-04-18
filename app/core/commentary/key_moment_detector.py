from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.models.Move import Move

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority tiers – lower number = higher priority.
# When multiple triggers fire on the same move the highest-priority one wins.
# ---------------------------------------------------------------------------
KEY_MOMENT_PRIORITY: Dict[str, int] = {
    "brilliant": 1,
    "blunder": 1,
    "critical_decision": 2,
    "structural_transformation": 2,
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


class KeyMomentDetector:
    """Detects key moments in a chess game.

    Returns the single highest-priority key moment type for a given move, or
    ``None`` if no trigger fires.
    """

    def __init__(self) -> None:
        self._prev_phase: Optional[str] = None
        self._prev_pawn_structure_type: Optional[str] = None
        self._last_book_depth: int = 0  # ply of last detected opening/book move

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        current_move: Move,
        previous_move: Optional[Move],
        pvs_for_move: Optional[List[List[Move]]],
    ) -> Optional[str]:
        """Return the highest-priority key moment type, or *None*."""
        candidates: List[str] = []

        # --- Score-based triggers (require both moves to have scores) ---
        if previous_move and current_move.score is not None and previous_move.score is not None:
            candidates.extend(self._score_based(current_move, previous_move, pvs_for_move))

        # --- Feature / strategic triggers ---
        candidates.extend(self._strategic_triggers(current_move, previous_move, pvs_for_move))

        if not candidates:
            # Update tracking state even when no trigger fires
            self._update_tracking(current_move)
            return None

        # Pick the highest-priority (lowest number) candidate
        best = min(candidates, key=lambda c: KEY_MOMENT_PRIORITY.get(c, 99))
        logger.info(f"Move {current_move.depth}: key moment = {best} (candidates: {candidates})")

        self._update_tracking(current_move)
        return best

    # ------------------------------------------------------------------
    # Score-based triggers
    # ------------------------------------------------------------------

    def _score_based(
        self,
        current_move: Move,
        previous_move: Move,
        pvs_for_move: Optional[List[List[Move]]],
    ) -> List[str]:
        results: List[str] = []

        is_white_move = current_move.depth % 2 == 1
        score_change = current_move.score - previous_move.score  # type: ignore[operator]

        # Adjust for mover's perspective
        perspective_change = score_change if is_white_move else -score_change

        log_msg = (
            f"Move {current_move.depth} ({'White' if is_white_move else 'Black'}): "
            f"Prev={previous_move.score}, Curr={current_move.score}, "
            f"PerspChange={perspective_change}"
        )
        logger.info(log_msg)

        # -- Blunder --
        if perspective_change <= -200:
            results.append("blunder")

        # -- Mistake --
        elif perspective_change <= -100:
            results.append("mistake")

        # -- Inaccuracy --
        elif perspective_change <= -50:
            results.append("inaccuracy")

        # -- Good defense: under pressure but score stabilises or improves --
        prev_eval = previous_move.score if is_white_move else -previous_move.score  # type: ignore[operator]
        if prev_eval is not None and prev_eval <= -100 and perspective_change >= 0:
            results.append("good_defense")

        # -- Missed opportunity (PV-based) --
        if pvs_for_move and pvs_for_move[0]:
            best_move_in_pv = pvs_for_move[0][0]
            if best_move_in_pv and best_move_in_pv.score is not None:
                opportunity_diff = best_move_in_pv.score - current_move.score  # type: ignore[operator]
                if not is_white_move:
                    opportunity_diff = -opportunity_diff
                if opportunity_diff >= 200:
                    results.append("missed_opportunity")

        # -- Brilliant: best move + material sacrifice + position improves --
        if pvs_for_move and pvs_for_move[0]:
            played_is_best = (
                pvs_for_move[0][0] and getattr(pvs_for_move[0][0], "move", None) == current_move.move
            )
            if played_is_best and perspective_change >= 0:
                material_before = self._get_material_diff(previous_move)
                material_after = self._get_material_diff(current_move)
                if material_before is not None and material_after is not None:
                    mat_change = material_after - material_before
                    if not is_white_move:
                        mat_change = -mat_change
                    if mat_change < -50:  # sacrificed material
                        results.append("brilliant")

        # -- Great move: best move + significantly better than 2nd best --
        if pvs_for_move and len(pvs_for_move) >= 2:
            pv1_first = pvs_for_move[0][0] if pvs_for_move[0] else None
            pv2_first = pvs_for_move[1][0] if pvs_for_move[1] else None
            played_is_best = (
                pv1_first and getattr(pv1_first, "move", None) == current_move.move
            )
            if played_is_best and pv1_first and pv2_first:
                s1 = pv1_first.score
                s2 = pv2_first.score
                if s1 is not None and s2 is not None:
                    gap = s1 - s2 if is_white_move else s2 - s1
                    if gap >= 80:
                        results.append("great_move")

        # -- Critical decision: top-2 PVs within 20cp but different plans --
        if pvs_for_move and len(pvs_for_move) >= 2:
            pv1_first = pvs_for_move[0][0] if pvs_for_move[0] else None
            pv2_first = pvs_for_move[1][0] if pvs_for_move[1] else None
            if pv1_first and pv2_first and pv1_first.score is not None and pv2_first.score is not None:
                gap = abs(pv1_first.score - pv2_first.score)
                if gap <= 20:
                    # Check if they lead to different pawn structures
                    ps1 = self._pawn_structure_type(pv1_first)
                    ps2 = self._pawn_structure_type(pv2_first)
                    if ps1 and ps2 and ps1 != ps2:
                        results.append("critical_decision")

        return results

    # ------------------------------------------------------------------
    # Strategic / feature-based triggers
    # ------------------------------------------------------------------

    def _strategic_triggers(
        self,
        current_move: Move,
        previous_move: Optional[Move],
        pvs_for_move: Optional[List[List[Move]]],
    ) -> List[str]:
        results: List[str] = []
        curr_hf = current_move.hiddenFeatures or {}
        prev_hf = (previous_move.hiddenFeatures if previous_move else None) or {}

        is_white = current_move.depth % 2 == 1
        side = "white" if is_white else "black"

        # -- Structural transformation: pawn structure type changed --
        curr_ps = (curr_hf.get("pawnStructure") or {}).get("centerType")
        if self._prev_pawn_structure_type and curr_ps and curr_ps != self._prev_pawn_structure_type:
            results.append("structural_transformation")

        # -- King safety crisis: exposure score jumps >= 3 --
        curr_exposure = (curr_hf.get(side) or {}).get("kingExposure")
        prev_exposure = (prev_hf.get(side) or {}).get("kingExposure")
        if curr_exposure is not None and prev_exposure is not None:
            if curr_exposure - prev_exposure >= 3:
                results.append("king_safety_crisis")

        # -- Initiative shift: mobility delta >= 8 AND attacking pieces delta >= 2 --
        curr_mobility = (curr_hf.get(side) or {}).get("mobility")
        prev_mobility = (prev_hf.get(side) or {}).get("mobility")
        curr_attacking = (curr_hf.get(side) or {}).get("attackingPieces")
        prev_attacking = (prev_hf.get(side) or {}).get("attackingPieces")
        if all(v is not None for v in [curr_mobility, prev_mobility, curr_attacking, prev_attacking]):
            mob_delta = curr_mobility - prev_mobility  # type: ignore[operator]
            atk_delta = curr_attacking - prev_attacking  # type: ignore[operator]
            if mob_delta >= 8 and atk_delta >= 2:
                results.append("initiative_shift")

        # -- Piece activation: centralization improves >= 3 --
        curr_cent = (curr_hf.get(side) or {}).get("centralization")
        prev_cent = (prev_hf.get(side) or {}).get("centralization")
        if curr_cent is not None and prev_cent is not None:
            if curr_cent - prev_cent >= 3:
                results.append("piece_activation")

        # -- Opening transition: first move out of book --
        # Heuristic: if previous move had an opening name in trace but current doesn't
        if previous_move and previous_move.trace and isinstance(previous_move.trace, dict):
            had_opening = previous_move.trace.get("openingName") or previous_move.trace.get("openingCode")
            if had_opening:
                curr_trace = current_move.trace if isinstance(current_move.trace, dict) else {}
                has_opening = curr_trace.get("openingName") or curr_trace.get("openingCode")
                if not has_opening and current_move.depth > self._last_book_depth:
                    results.append("opening_transition")
                    self._last_book_depth = current_move.depth

        # -- Endgame transition: phase changes from mid to end --
        if self._prev_phase and current_move.phase:
            if self._prev_phase in ("opening", "mid", "middlegame") and current_move.phase in ("end", "endgame"):
                results.append("endgame_transition")

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_tracking(self, move: Move) -> None:
        """Update internal tracking state after processing a move."""
        self._prev_phase = move.phase
        hf = move.hiddenFeatures or {}
        ps = hf.get("pawnStructure")
        if isinstance(ps, dict):
            self._prev_pawn_structure_type = ps.get("centerType")

    @staticmethod
    def _get_material_diff(move: Move) -> Optional[float]:
        """Get total material difference from hiddenFeatures."""
        hf = move.hiddenFeatures
        if not isinstance(hf, dict):
            return None
        material = hf.get("material")
        if not isinstance(material, dict):
            return None
        diff = material.get("diff")
        if not isinstance(diff, dict):
            return None
        return diff.get("total")

    @staticmethod
    def _pawn_structure_type(move: Move) -> Optional[str]:
        """Extract pawn structure center type from a move's hiddenFeatures."""
        hf = move.hiddenFeatures
        if not isinstance(hf, dict):
            return None
        ps = hf.get("pawnStructure")
        if not isinstance(ps, dict):
            return None
        return ps.get("centerType")
