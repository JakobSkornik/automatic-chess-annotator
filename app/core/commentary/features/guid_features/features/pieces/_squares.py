"""Outpost-square detection, ported from the Stockfish guide.

An outpost square (``outpost_square``) is a square on relative ranks 4-6 that a
friendly pawn defends from behind and that no enemy pawn can ever attack
(``pawn_attacks_span``). These predicates back the knight/bishop outpost feature.
"""

from __future__ import annotations

import chess

from .._board import forward_step, is_pawn, relative_rank
from ..pawns.backward import is_backward

_OUTPOST_MIN_RANK = 3  # relative rank 4 (0-based)
_OUTPOST_MAX_RANK = 5  # relative rank 6 (0-based)


def pawn_attacks_span(board: chess.Board, square: int, color: chess.Color) -> bool:
    """Whether an enemy pawn currently attacks, or can advance to attack, the
    square (Stockfish ``pawn_attacks_span``). A backward or blocked enemy pawn
    that can never get there does not count."""
    enemy = not color
    fw = forward_step(color)
    file, rank = chess.square_file(square), chess.square_rank(square)
    scan = rank + fw
    while 0 <= scan < 8:
        for adj_file in (file - 1, file + 1):
            if not 0 <= adj_file < 8:
                continue
            if not is_pawn(board, adj_file, scan, enemy):
                continue
            attacks_directly = scan == rank + fw
            blocked_by_us = is_pawn(board, adj_file, scan - fw, color)
            if attacks_directly or (
                not blocked_by_us
                and not is_backward(board, chess.square(adj_file, scan), enemy)
            ):
                return True
        scan += fw
    return False


def is_outpost_square(board: chess.Board, square: int, color: chess.Color) -> bool:
    """Whether ``square`` is an outpost square for ``color``."""
    rank = relative_rank(square, color)
    if rank < _OUTPOST_MIN_RANK or rank > _OUTPOST_MAX_RANK:
        return False
    file = chess.square_file(square)
    behind = -forward_step(color)
    base_rank = chess.square_rank(square)
    pawn_supported = is_pawn(board, file - 1, base_rank + behind, color) or is_pawn(
        board, file + 1, base_rank + behind, color
    )
    if not pawn_supported:
        return False
    return not pawn_attacks_span(board, square, color)
