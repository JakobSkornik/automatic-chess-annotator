"""Backward pawns (Stockfish ``backward``).

A pawn is backward when no friendly pawn on an adjacent file stands level with
or behind it (so it cannot be supported in advancing), and an enemy pawn guards
or blocks the square in front of it.
"""

from __future__ import annotations

import chess

from .._board import forward_step, is_pawn


def _supported_by_neighbour(
    board: chess.Board, file: int, rank: int, color: chess.Color
) -> bool:
    """A friendly pawn on an adjacent file, level with or behind this pawn."""
    behind = -forward_step(color)
    scan = rank
    while 0 <= scan < 8:
        if is_pawn(board, file - 1, scan, color) or is_pawn(
            board, file + 1, scan, color
        ):
            return True
        scan += behind
    return False


def _advance_is_contested(
    board: chess.Board, file: int, rank: int, color: chess.Color
) -> bool:
    """An enemy pawn guards the stop square in front, or blocks it directly."""
    enemy = not color
    fw = forward_step(color)
    stop_rank = rank + fw
    guard_rank = rank + 2 * fw
    return (
        is_pawn(board, file - 1, guard_rank, enemy)
        or is_pawn(board, file + 1, guard_rank, enemy)
        or is_pawn(board, file, stop_rank, enemy)
    )


def is_backward(board: chess.Board, square: int, color: chess.Color) -> bool:
    """Whether the pawn of ``color`` on ``square`` is backward."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type != chess.PAWN or piece.color != color:
        return False
    file, rank = chess.square_file(square), chess.square_rank(square)
    if _supported_by_neighbour(board, file, rank, color):
        return False
    return _advance_is_contested(board, file, rank, color)


def count(board: chess.Board, color: chess.Color) -> int:
    """Number of backward pawns for ``color``."""
    return sum(
        1
        for square in board.pieces(chess.PAWN, color)
        if is_backward(board, square, color)
    )
