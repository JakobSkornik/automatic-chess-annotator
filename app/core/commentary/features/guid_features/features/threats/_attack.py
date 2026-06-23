"""Attack/defence helpers shared by the threat predicates.

Stockfish's threat features compare how many of our pieces attack a square with
how many enemy pieces defend it. We use python-chess attacker sets (direct
attackers; slider x-rays are not counted — a close, simpler approximation).
"""

from __future__ import annotations

import chess


def attacker_count(board: chess.Board, color: chess.Color, square: int) -> int:
    """Number of ``color`` pieces attacking ``square``."""
    return len(board.attackers(color, square))


def is_pawn_defended(board: chess.Board, square: int, color: chess.Color) -> bool:
    """Whether a pawn of ``color`` defends ``square``."""
    return any(
        board.piece_type_at(attacker) == chess.PAWN
        for attacker in board.attackers(color, square)
    )
