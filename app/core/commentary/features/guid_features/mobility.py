"""Piece mobility, ported from the Stockfish Evaluation Guide.

Mobility = the number of squares a side's minor/major pieces can reach inside
the *mobility area* (Stockfish's definition: squares not controlled by enemy
pawns, not holding our blocked/early pawns, king, queen, or pinned pieces).

Unlike Stockfish we keep only the structural COUNT (a natural, explainable unit
— "squares the pieces can reach"), not the tuned mg/eg centipawn bonus: the
position's evaluation comes from the engine, so the bonus table is not needed.
"""

from __future__ import annotations

import chess

_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_QUEEN_DIRS = _BISHOP_DIRS + _ROOK_DIRS


def _enemy_pawn_attacks(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    bb = chess.SquareSet()
    for sq in board.pieces(chess.PAWN, not color):
        bb |= board.attacks(sq)
    return bb


def _mobility_area(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    """Squares that count toward ``color``'s mobility (Stockfish ``mobility_area``):
    exclude our king/queen squares, squares attacked by enemy pawns, our pawns on
    ranks 1-3 or blocked from advancing, and our pinned pieces (blockers for the
    king)."""
    enemy_pawn_atk = _enemy_pawn_attacks(board, color)
    own_king = board.king(color)
    area = chess.SquareSet()
    forward = 8 if color == chess.WHITE else -8
    for sq in chess.SQUARES:
        if sq == own_king or sq in enemy_pawn_atk:
            continue
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            if piece.piece_type == chess.QUEEN:
                continue
            if piece.piece_type == chess.PAWN:
                rel_rank = (
                    chess.square_rank(sq)
                    if color == chess.WHITE
                    else 7 - chess.square_rank(sq)
                )
                ahead = sq + forward
                blocked = 0 <= ahead < 64 and board.piece_at(ahead) is not None
                if rel_rank < 3 or blocked:
                    continue
            if board.is_pinned(color, sq):
                continue
        area.add(sq)
    return area


def _slider_attacks(board: chess.Board, sq: int, dirs, transparent) -> chess.SquareSet:
    """Squares reachable along ``dirs``, x-raying through pieces for which
    ``transparent(piece)`` is true (the blocking square itself is still counted)."""
    f0, r0 = chess.square_file(sq), chess.square_rank(sq)
    out = chess.SquareSet()
    for df, dr in dirs:
        f, r = f0 + df, r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            target = chess.square(f, r)
            out.add(target)
            blocker = board.piece_at(target)
            if blocker is not None and not transparent(blocker):
                break
            f += df
            r += dr
    return out


def _piece_mobility(
    board: chess.Board, sq: int, color: chess.Color, area: chess.SquareSet
) -> int:
    piece = board.piece_at(sq)
    if piece is None:
        return 0
    own_queens = board.pieces(chess.QUEEN, color)
    pt = piece.piece_type

    if pt == chess.KNIGHT:
        if board.is_pinned(color, sq):  # a pinned knight cannot move at all
            return 0
        att = chess.SquareSet(board.attacks(sq)) - own_queens
        return len(att & area)
    if pt == chess.BISHOP:
        att = _slider_attacks(
            board, sq, _BISHOP_DIRS, lambda p: p.piece_type == chess.QUEEN
        )
        att &= board.pin(color, sq)  # BB_ALL when not pinned
        att -= own_queens
        return len(att & area)
    if pt == chess.ROOK:
        att = _slider_attacks(
            board,
            sq,
            _ROOK_DIRS,
            lambda p: (
                p.piece_type == chess.QUEEN
                or (p.piece_type == chess.ROOK and p.color == color)
            ),
        )
        att &= board.pin(color, sq)
        return len(att & area)
    if pt == chess.QUEEN:
        att = _slider_attacks(board, sq, _QUEEN_DIRS, lambda p: False)
        att &= board.pin(color, sq)
        return len(att & area)
    return 0


def mobility_count(board: chess.Board, color: chess.Color) -> int:
    """Total reachable squares (in the mobility area) for ``color``'s N/B/R/Q."""
    area = _mobility_area(board, color)
    total = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            total += _piece_mobility(board, sq, color, area)
    return total
