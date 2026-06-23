"""The rule context (``_Ctx``) every rule reads, plus side-perspective helpers."""

from __future__ import annotations

from collections.abc import Callable

import chess

from app.models.comment_facts import Claim, FeatureDelta, FeatureDiff

from .constants import COUNT_CLAIM_CP


def _toward(side: str, white_pov_delta: int) -> int:
    """Positive = good for ``side``."""
    return white_pov_delta if side == "WHITE" else -white_pov_delta


def _side_label(side: str) -> str:
    return "White" if side == "WHITE" else "Black"


def _benef(side: str) -> str:
    """Beneficiary key for a claim that favors ``side``."""
    return side.lower()


def _benef_opp(side: str) -> str:
    """Beneficiary key for a claim that favors the opponent of ``side``."""
    return "black" if side == "WHITE" else "white"


class _Ctx:
    """Everything a rule may look at."""

    def __init__(
        self,
        diff: FeatureDiff,
        phase: str,
        mover: str,
        eval_cp: int | None,
        start_board: chess.Board | None = None,
        leaf_board: chess.Board | None = None,
    ) -> None:
        self.phase = phase
        self.mover = mover  # "WHITE" | "BLACK"
        self.eval_cp = eval_cp
        # Boards behind the diff: rules use them to name squares and files
        # ("passed pawn on e5") — deterministic, no LLM involved.
        self.start_board = start_board
        self.leaf_board = leaf_board
        self.by_name: dict[str, FeatureDelta] = {}
        for d in list(diff.positive) + list(diff.negative):
            self.by_name[d.name] = d

    def color(self, side: str) -> chess.Color:
        return chess.WHITE if side == "WHITE" else chess.BLACK

    def new_squares(self, lookup, side: str) -> list[str]:
        """Squares satisfying `lookup` on the leaf board but not at the start."""
        if self.leaf_board is None:
            return []
        leaf = lookup(self.leaf_board, self.color(side))
        if self.start_board is None:
            return leaf
        start = set(lookup(self.start_board, self.color(side)))
        return [s for s in leaf if s not in start]

    def delta(self, name: str) -> int:
        d = self.by_name.get(name)
        return d.delta_cp if d else 0

    def claim_cp(self, name: str) -> int:
        """Magnitude of a feature's change as a cp-comparable claim importance.

        Centipawn features pass through; natural-count features (mobility, pawn
        counts) are scaled by their per-unit weight so claims rank consistently."""
        base = name.removeprefix("WHITE_").removeprefix("BLACK_")
        return abs(self.delta(name)) * COUNT_CLAIM_CP.get(base, 1)

    def flag_change(self, name: str) -> str | None:
        d = self.by_name.get(name)
        if d is None or d.flag_before is None or d.flag_after is None:
            return None
        if d.flag_before == d.flag_after:
            return None
        return f"{d.flag_before} -> {d.flag_after}"

    def flag_pair(self, name: str) -> tuple | None:
        d = self.by_name.get(name)
        if d is None or d.flag_before is None or d.flag_after is None:
            return None
        return (d.flag_before, d.flag_after)


Rule = Callable[[_Ctx], list[Claim]]
