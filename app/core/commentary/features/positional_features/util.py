"""Constants and leaf board-geometry helpers shared across the feature modules."""

from __future__ import annotations

import chess

FILES = [[chess.square(file_idx, rank) for rank in range(8)] for file_idx in range(8)]

CENTER_SQUARES = [
    chess.D4,
    chess.E4,
    chess.D5,
    chess.E5,
]

FILE_NAMES = "abcdefgh"
# Endgame heuristic: fewer than this many minor+major pieces on the board.
ENDGAME_MINOR_MAJOR_MAX = 7
CONTESTED_SQUARES_CAP = 10

CASTLING_KEYS = frozenset({"canCastleKingSide", "canCastleQueenSide"})
PROTECTED_TOP_LEVEL_KEYS = frozenset({"material", "pawnStructure", "openFiles"})
KEEP_ZERO_METRICS = frozenset(
    {
        "mobility",
        "centerControl",
        "kingShieldPawns",
        "openFilesAdjacent",
        "semiOpenFilesAdjacent",
        "kingZoneAttacks",
        "attackedPieces",
        "attackingPieces",
        "rooksOnOpenFiles",
        "rooksOnSemiOpenFiles",
        "space",
        "lightSquareBishops",
        "darkSquareBishops",
        "goodBishops",
        "badBishops",
        "kingExposure",
        "centralization",
    }
)


def _sq_name(sq: int) -> str:
    return chess.square_name(sq)


def _file_index(square: int) -> int:
    return chess.square_file(square)


def _is_light_square(square: int) -> bool:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    return (file_idx + rank_idx) % 2 == 0


def _pawn_files(board: chess.Board, color: chess.Color) -> dict[int, list[int]]:
    files: dict[int, list[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        fi = _file_index(sq)
        files.setdefault(fi, []).append(sq)
    for fi in files:
        files[fi].sort(key=lambda s: chess.square_rank(s))
    return files


def _is_passed_pawn(board: chess.Board, pawn_sq: int, color: chess.Color) -> bool:
    pawn_file = chess.square_file(pawn_sq)
    enemy = not color
    ranks = (
        range(chess.square_rank(pawn_sq) + 1, 8)
        if color == chess.WHITE
        else range(chess.square_rank(pawn_sq) - 1, -1, -1)
    )
    for rank in ranks:
        for f in (pawn_file - 1, pawn_file, pawn_file + 1):
            if 0 <= f <= 7:
                sq = chess.square(f, rank)
                if sq in board.pieces(chess.PAWN, enemy):
                    return False
    return True


def _side_squares(board: chess.Board, color: chess.Color) -> set[int]:
    squares: set[int] = set()
    for pt in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ):
        squares.update(board.pieces(pt, color))
    return squares


def _minor_major_count(board: chess.Board) -> int:
    s = 0
    for c in (chess.WHITE, chess.BLACK):
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            s += len(board.pieces(pt, c))
    return s


def _is_weak_square(board: chess.Board, sq: int, for_color: chess.Color) -> bool:
    defender = for_color
    sq_file = chess.square_file(sq)
    sq_rank = chess.square_rank(sq)
    for f in (sq_file - 1, sq_file + 1):
        if not 0 <= f <= 7:
            continue
        if defender == chess.WHITE:
            for r in range(1, sq_rank):
                pawn_sq = chess.square(f, r)
                if pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
                    return False
        else:
            for r in range(sq_rank + 1, 7):
                pawn_sq = chess.square(f, r)
                if pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
                    return False
    return True


def _clear_between(board: chess.Board, a: int, b: int) -> bool:
    if a == b:
        return True
    file_a, ra = chess.square_file(a), chess.square_rank(a)
    file_b, rb = chess.square_file(b), chess.square_rank(b)
    df = (file_b > file_a) - (file_b < file_a)
    dr = (rb > ra) - (rb < ra)
    if df != 0 and dr != 0 and abs(file_b - file_a) != abs(rb - ra):
        return False
    f, r = file_a + df, ra + dr
    while f != file_b or r != rb:
        sq = chess.square(f, r)
        if board.piece_at(sq) is not None:
            return False
        f += df
        r += dr
    return True
