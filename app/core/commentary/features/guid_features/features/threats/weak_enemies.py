"""Weak enemies (Stockfish ``weak_enemies``).

An enemy piece is weak when it is not defended by an enemy pawn and is under our
attack — unless we attack it only once while it is defended more than once.
"""

from __future__ import annotations

import chess

from ._attack import attacker_count, is_pawn_defended


def is_weak_enemy(board: chess.Board, square: int, attacker: chess.Color) -> bool:
    """Whether the enemy piece on ``square`` is weak from ``attacker``'s side."""
    enemy = not attacker
    piece = board.piece_at(square)
    if piece is None or piece.color != enemy:
        return False
    if is_pawn_defended(board, square, enemy):
        return False
    attacks = attacker_count(board, attacker, square)
    if attacks == 0:
        return False
    over_defended = attacks <= 1 and attacker_count(board, enemy, square) > 1
    return not over_defended


def count(board: chess.Board, attacker: chess.Color) -> int:
    """Number of weak enemy pieces from ``attacker``'s side."""
    enemy = not attacker
    return sum(
        1
        for square in chess.SquareSet(board.occupied_co[enemy])
        if is_weak_enemy(board, square, attacker)
    )
