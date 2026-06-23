"""Hanging pieces (Stockfish ``hanging``).

A weak enemy that is either undefended, or a non-pawn we attack more than once.
"""

from __future__ import annotations

import chess

from ._attack import attacker_count
from .weak_enemies import is_weak_enemy


def is_hanging(board: chess.Board, square: int, attacker: chess.Color) -> bool:
    """Whether the enemy piece on ``square`` hangs to ``attacker``."""
    if not is_weak_enemy(board, square, attacker):
        return False
    enemy = not attacker
    piece = board.piece_at(square)
    if piece.piece_type != chess.PAWN and attacker_count(board, attacker, square) > 1:
        return True
    return attacker_count(board, enemy, square) == 0


def count(board: chess.Board, attacker: chess.Color) -> int:
    """Number of hanging enemy pieces from ``attacker``'s side."""
    enemy = not attacker
    return sum(
        1
        for square in chess.SquareSet(board.occupied_co[enemy])
        if is_hanging(board, square, attacker)
    )
