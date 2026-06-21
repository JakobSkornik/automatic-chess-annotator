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


def _is_isolated(files: dict[int, list[int]], sq: int) -> bool:
    f = chess.square_file(sq)
    return not files.get(f - 1) and not files.get(f + 1)


def _is_backward(board: chess.Board, sq: int, color: chess.Color) -> bool:
    """No friendly pawn on an adjacent file at or behind it, and its stop
    square is controlled by an enemy pawn."""
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    behind_ok = False
    for nsq in board.pieces(chess.PAWN, color):
        nf = chess.square_file(nsq)
        if abs(nf - f) != 1:
            continue
        nr = chess.square_rank(nsq)
        if (color == chess.WHITE and nr <= r) or (color == chess.BLACK and nr >= r):
            behind_ok = True
            break
    if behind_ok:
        return False
    step = 1 if color == chess.WHITE else -1
    stop_r = r + step
    if stop_r < 0 or stop_r > 7:
        return False
    enemy = not color
    # enemy pawn attack on the stop square
    for df in (-1, 1):
        ef = f + df
        if ef < 0 or ef > 7:
            continue
        er = stop_r + step
        if er < 0 or er > 7:
            continue
        ep = board.piece_at(chess.square(ef, er))
        if ep is not None and ep.piece_type == chess.PAWN and ep.color == enemy:
            return True
    return False


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
