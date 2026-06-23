"""Pinned enemy pieces (Stockfish ``pinned`` / ``blockers_for_king``).

Counts the enemy minor/major pieces absolutely pinned to their king — i.e. ones
our sliders hold in place. A pin is a concrete, pragmatic asset, so it is worth
stating even when the evaluation is already decided.
"""

from __future__ import annotations

import chess

_PINNABLE = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def count(board: chess.Board, color: chess.Color) -> int:
    """Number of enemy pieces ``color`` pins to the enemy king."""
    enemy = not color
    return sum(
        1
        for piece_type in _PINNABLE
        for square in board.pieces(piece_type, enemy)
        if board.is_pinned(enemy, square)
    )
