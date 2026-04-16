"""Tracks the narrative arc of a chess game across moves.

Maintains a running context (key moments history, score trend, turning points,
structural changes, opening info) that is included in every LLM call so the
model can produce commentary aware of the game's story so far.

Typical cost: ~50-100 tokens per call when serialised via ``get_narrative_context()``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class NarrativeTracker:
    """Accumulates game context across moves for richer LLM commentary."""

    def __init__(self) -> None:
        # Chronological list of key moments: {"ply": int, "type": str, "side": str}
        self.key_moments: List[Dict[str, Any]] = []
        # Recent scores (white-POV centipawns) – last N entries
        self.score_trend: List[float] = []
        # Ply numbers where the advantage changed sides
        self.turning_points: List[int] = []
        # Pawn structure center-type per ply (to detect changes)
        self.structural_history: List[Optional[str]] = []
        # Opening name detected so far
        self.opening_name: str = ""
        self.opening_eco: str = ""
        # Which side currently has the advantage (or "equal")
        self._advantage_side: str = "equal"
        # Maximum trend length kept
        self._max_trend = 8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        ply: int,
        score: Optional[float],
        key_moment_type: Optional[str],
        hidden_features: Optional[Dict[str, Any]],
        opening: Optional[Dict[str, str]] = None,
    ) -> None:
        """Call once per analysed move to update the narrative state."""
        is_white = ply % 2 == 1
        side = "White" if is_white else "Black"

        # Opening
        if opening:
            name = opening.get("name")
            eco = opening.get("eco")
            if name:
                self.opening_name = name
            if eco:
                self.opening_eco = eco

        # Score trend
        if score is not None:
            self.score_trend.append(score)
            if len(self.score_trend) > self._max_trend:
                self.score_trend = self.score_trend[-self._max_trend:]

            # Detect turning point (advantage changes side)
            new_side = "White" if score > 50 else ("Black" if score < -50 else "equal")
            if new_side != self._advantage_side and new_side != "equal" and self._advantage_side != "equal":
                self.turning_points.append(ply)
            self._advantage_side = new_side

        # Key moment
        if key_moment_type:
            self.key_moments.append({"ply": ply, "type": key_moment_type, "side": side})

        # Structural history
        ps = None
        if isinstance(hidden_features, dict):
            ps_dict = hidden_features.get("pawnStructure")
            if isinstance(ps_dict, dict):
                ps = ps_dict.get("centerType")
        self.structural_history.append(ps)

    def get_narrative_context(self) -> str:
        """Return a concise 2-4 sentence narrative summary for LLM consumption."""
        parts: List[str] = []

        # Opening info
        if self.opening_name:
            eco_tag = f" ({self.opening_eco})" if self.opening_eco else ""
            parts.append(f"Opening: {self.opening_name}{eco_tag}.")

        # Advantage summary
        if self.score_trend:
            latest = self.score_trend[-1]
            if latest > 150:
                parts.append("White has a significant advantage.")
            elif latest > 50:
                parts.append("White has a slight edge.")
            elif latest < -150:
                parts.append("Black has a significant advantage.")
            elif latest < -50:
                parts.append("Black has a slight edge.")
            else:
                parts.append("The position is roughly equal.")

        # Turning points
        if self.turning_points:
            tp_strs = [f"move {(p + 1) // 2}" for p in self.turning_points[-3:]]
            parts.append(f"Turning points at {', '.join(tp_strs)}.")

        # Key moments summary
        if self.key_moments:
            recent = self.key_moments[-4:]
            moment_strs = [
                f"{m['side']} {m['type'].replace('_', ' ')} on move {(m['ply'] + 1) // 2}"
                for m in recent
            ]
            parts.append(f"Key moments: {'; '.join(moment_strs)}.")

        # Structural changes
        unique_structures = []
        for s in self.structural_history:
            if s and (not unique_structures or unique_structures[-1] != s):
                unique_structures.append(s)
        if len(unique_structures) >= 2:
            parts.append(
                f"Pawn structure evolved: {' -> '.join(unique_structures[-3:])}."
            )

        return " ".join(parts) if parts else "Game in progress."
