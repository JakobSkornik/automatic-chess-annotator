"""Pawn-structure feature helpers (files, doubled/isolated/backward/passed, islands)."""

from __future__ import annotations

from typing import Any

import chess

from .util import (
    FILE_NAMES,
    FILES,
    _is_passed_pawn,
    _pawn_files,
    _sq_name,
)


def _open_and_semi_open_files(board: chess.Board) -> dict[str, list[int]]:
    open_files: list[int] = []
    semi_open_white: list[int] = []
    semi_open_black: list[int] = []
    for f in range(8):
        has_white_pawn = any(
            sq in board.pieces(chess.PAWN, chess.WHITE) for sq in FILES[f]
        )
        has_black_pawn = any(
            sq in board.pieces(chess.PAWN, chess.BLACK) for sq in FILES[f]
        )
        if not has_white_pawn and not has_black_pawn:
            open_files.append(f)
        else:
            if not has_white_pawn and has_black_pawn:
                semi_open_white.append(f)
            if not has_black_pawn and has_white_pawn:
                semi_open_black.append(f)
    return {
        "open": open_files,
        "semiOpenWhite": semi_open_white,
        "semiOpenBlack": semi_open_black,
    }


def _pawn_structure_classification(board: chess.Board) -> dict[str, object]:
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    center_files = {2, 3, 4, 5}
    white_center = [
        sq
        for sq in white_pawns
        if chess.square_file(sq) in center_files and 2 <= chess.square_rank(sq) <= 5
    ]
    black_center = [
        sq
        for sq in black_pawns
        if chess.square_file(sq) in center_files and 2 <= chess.square_rank(sq) <= 5
    ]

    tension: list[str] = []
    for wp in white_pawns:
        wf, wr = chess.square_file(wp), chess.square_rank(wp)
        for df in (-1, 1):
            target_f = wf + df
            target_r = wr + 1
            if 0 <= target_f <= 7 and 0 <= target_r <= 7:
                target_sq = chess.square(target_f, target_r)
                if target_sq in black_pawns:
                    tension.append(f"{_sq_name(wp)}-{_sq_name(target_sq)}")

    locked_count = 0
    for wp in white_center:
        wf, wr = chess.square_file(wp), chess.square_rank(wp)
        front_sq = chess.square(wf, wr + 1)
        if front_sq in black_pawns:
            locked_count += 1

    if locked_count >= 2:
        center_type = "closed"
    elif not white_center and not black_center:
        center_type = "open"
    else:
        center_type = "semi-open"

    breaks: list[str] = []
    for color, own_pawns_set, enemy_pawns_set, forward in [
        (chess.WHITE, white_pawns, black_pawns, 1),
        (chess.BLACK, black_pawns, white_pawns, -1),
    ]:
        side = "White" if color == chess.WHITE else "Black"
        for pawn_sq in own_pawns_set:
            pf, pr = chess.square_file(pawn_sq), chess.square_rank(pawn_sq)
            advance_sq = (
                chess.square(pf, pr + forward) if 0 <= pr + forward <= 7 else None
            )
            if advance_sq is None:
                continue
            if board.piece_at(advance_sq) is not None:
                continue
            for df in (-1, 1):
                diag_f = pf + df
                diag_r = pr + forward + forward
                if not (0 <= diag_f <= 7 and 0 <= diag_r <= 7):
                    continue
                diag_sq = chess.square(diag_f, diag_r)
                if diag_sq in enemy_pawns_set:
                    breaks.append(f"{side} {_sq_name(pawn_sq)}-{_sq_name(advance_sq)}")
                    break
    return {
        "centerType": center_type,
        "tension": tension[:6],
        "breaks": breaks[:6],
    }


def _doubled_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    pf = _pawn_files(board, color)
    out: list[str] = []
    for lst in pf.values():
        if len(lst) > 1:
            out.extend(_sq_name(s) for s in lst)
    return sorted(set(out), key=lambda n: (n[1], n[0]))


def _isolated_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    pf = _pawn_files(board, color)
    out: list[str] = []
    for f, lst in pf.items():
        has_left = (f - 1) in pf
        has_right = (f + 1) in pf
        if not has_left and not has_right:
            out.extend(_sq_name(s) for s in lst)
    return out


def _passed_pawn_records(
    board: chess.Board, color: chess.Color
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sq in board.pieces(chess.PAWN, color):
        if _is_passed_pawn(board, sq, color):
            human_rank = chess.square_rank(sq) + 1
            out.append({"sq": _sq_name(sq), "rank": human_rank})
    return sorted(out, key=lambda d: d["sq"])


def _backward_pawn_squares(board: chess.Board, color: chess.Color) -> list[str]:
    enemy = not color
    pf = _pawn_files(board, color)
    out: list[str] = []
    for pawn_sq in board.pieces(chess.PAWN, color):
        pr = chess.square_rank(pawn_sq)
        pf_i = chess.square_file(pawn_sq)
        forward = 1 if color == chess.WHITE else -1
        adv_r = pr + forward
        if not (0 <= adv_r <= 7):
            continue
        advance_sq = chess.square(pf_i, adv_r)
        blocked_or_attacked = board.piece_at(
            advance_sq
        ) is not None or board.is_attacked_by(enemy, advance_sq)
        if not blocked_or_attacked:
            continue
        has_advanced_neighbor = False
        for df in (-1, 1):
            nf = pf_i + df
            if not 0 <= nf <= 7:
                continue
            if nf not in pf:
                continue
            for osq in pf[nf]:
                orank = chess.square_rank(osq)
                if color == chess.WHITE and orank > pr:
                    has_advanced_neighbor = True
                    break
                if color == chess.BLACK and orank < pr:
                    has_advanced_neighbor = True
                    break
            if has_advanced_neighbor:
                break
        if not has_advanced_neighbor:
            out.append(_sq_name(pawn_sq))
    return out


def _pawn_islands(board: chess.Board, color: chess.Color) -> dict[str, Any]:
    pf = sorted(_pawn_files(board, color).keys())
    if not pf:
        return {"count": 0, "runs": []}
    runs: list[str] = []
    start = prev = pf[0]
    for f in pf[1:]:
        if f == prev + 1:
            prev = f
        else:
            runs.append(f"{FILE_NAMES[start]}-{FILE_NAMES[prev]}")
            start = prev = f
    runs.append(f"{FILE_NAMES[start]}-{FILE_NAMES[prev]}")
    return {"count": len(runs), "runs": runs}
