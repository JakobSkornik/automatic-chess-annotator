"""Pure board geometry / pawn-relation helpers shared by the feature vector."""

from __future__ import annotations

import chess


def _pawn_files(board: chess.Board, color: chess.Color) -> dict[int, list[int]]:
    """file -> list of pawn squares for ``color``."""
    files: dict[int, list[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        files.setdefault(chess.square_file(sq), []).append(sq)
    return files


def _is_passed(board: chess.Board, sq: int, color: chess.Color) -> bool:
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    enemy = not color
    for ef in (f - 1, f, f + 1):
        if ef < 0 or ef > 7:
            continue
        for esq in board.pieces(chess.PAWN, enemy):
            if chess.square_file(esq) != ef:
                continue
            er = chess.square_rank(esq)
            if color == chess.WHITE and er > r:
                return False
            if color == chess.BLACK and er < r:
                return False
    return True


def _relative_rank(sq: int, color: chess.Color) -> int:
    r = chess.square_rank(sq)
    return r if color == chess.WHITE else 7 - r


def _center_ring_bonus(sq: int) -> int:
    """0 on the rim, 3 in the four center squares (ring distance toward center)."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    df = min(f, 7 - f)
    dr = min(r, 7 - r)
    return min(df, dr)


def _chebyshev(a: int, b: int) -> int:
    return chess.square_distance(a, b)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
