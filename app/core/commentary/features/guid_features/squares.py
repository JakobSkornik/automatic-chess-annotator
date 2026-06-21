"""Square-level lookups for grounded claim texts ("passed pawn on e5", ...)."""

from __future__ import annotations

import chess

from .geometry import _is_passed, _pawn_files, _relative_rank


def passed_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    return [
        chess.square_name(sq)
        for sq in board.pieces(chess.PAWN, color)
        if _is_passed(board, sq, color)
    ]


def doubled_pawn_files(board: chess.Board, color: chess.Color) -> list[str]:
    files = _pawn_files(board, color)
    return [chess.FILE_NAMES[f] for f, sqs in sorted(files.items()) if len(sqs) > 1]


def outpost_squares(board: chess.Board, color: chess.Color) -> list[str]:
    out: list[str] = []
    for sq in board.pieces(chess.KNIGHT, color):
        rel = _relative_rank(sq, color)
        if not (3 <= rel <= 5):
            continue
        defended = any(
            (p := board.piece_at(att)) is not None
            and p.piece_type == chess.PAWN
            and p.color == color
            for att in board.attackers(color, sq)
        )
        if not defended:
            continue
        f, r = chess.square_file(sq), chess.square_rank(sq)
        assailable = any(
            abs(chess.square_file(esq) - f) == 1
            and (
                (color == chess.WHITE and chess.square_rank(esq) > r)
                or (color == chess.BLACK and chess.square_rank(esq) < r)
            )
            for esq in board.pieces(chess.PAWN, not color)
        )
        if not assailable:
            out.append(chess.square_name(sq))
    return out


def bad_bishop_squares(board: chess.Board, color: chess.Color) -> list[str]:
    out: list[str] = []
    pawns = list(board.pieces(chess.PAWN, color))
    for sq in board.pieces(chess.BISHOP, color):
        light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
        weighted = 0
        for p in pawns:
            if ((chess.square_file(p) + chess.square_rank(p)) % 2 == 1) != light:
                continue
            pf = chess.square_file(p)
            weighted += 3 if pf in (3, 4) else 2 if pf in (2, 5) else 1
        if weighted >= 7 and len(board.attacks(sq)) <= 3:
            out.append(chess.square_name(sq))
    return out


def rooks_on_seventh_squares(board: chess.Board, color: chess.Color) -> list[str]:
    return [
        chess.square_name(sq)
        for sq in board.pieces(chess.ROOK, color)
        if _relative_rank(sq, color) == 6
    ]
