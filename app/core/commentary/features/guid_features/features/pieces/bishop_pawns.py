"""Bishop vs own pawns (Stockfish ``bishop_pawns``).

For each bishop: the number of friendly pawns on its colour complex, multiplied
by one plus the number of friendly blocked pawns on the central files (c-f). An
undefended bishop counts for slightly more. Higher = the bishop is more hemmed in.
"""

from __future__ import annotations

import chess

_CENTRAL_FILES = (2, 3, 4, 5)  # c, d, e, f


def _is_light(square: int) -> bool:
    return (chess.square_file(square) + chess.square_rank(square)) % 2 == 1


def _blocked_central_pawns(board: chess.Board, color: chess.Color) -> int:
    forward = 8 if color == chess.WHITE else -8
    total = 0
    for square in board.pieces(chess.PAWN, color):
        if chess.square_file(square) not in _CENTRAL_FILES:
            continue
        ahead = square + forward
        if 0 <= ahead < 64 and board.piece_at(ahead) is not None:
            total += 1
    return total


def count(board: chess.Board, color: chess.Color) -> int:
    """Stockfish bishop-pawns score for ``color`` (summed over its bishops)."""
    blocked = _blocked_central_pawns(board, color)
    own_pawns = list(board.pieces(chess.PAWN, color))
    total = 0
    for bishop in board.pieces(chess.BISHOP, color):
        same_colour_pawns = sum(
            1 for pawn in own_pawns if _is_light(pawn) == _is_light(bishop)
        )
        defended_by_pawn = any(
            attacker in board.pieces(chess.PAWN, color)
            for attacker in board.attackers(color, bishop)
        )
        total += same_colour_pawns * (blocked + (0 if defended_by_pawn else 1))
    return total
