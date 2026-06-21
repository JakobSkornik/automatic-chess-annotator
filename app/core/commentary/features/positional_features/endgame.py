"""Endgame-specific feature helpers (passers, opposite bishops, mop-up, rule of square)."""

from __future__ import annotations

from typing import Any

import chess

from .util import _is_passed_pawn, _sq_name

ROUGH_MATERIAL_BALANCE_CP = (
    500  # |material| under this = roughly level (endgame heuristics)
)


def _blockades(board: chess.Board) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for color in (chess.WHITE, chess.BLACK):
        enemy = not color
        fwd = 1 if color == chess.WHITE else -1
        for psq in board.pieces(chess.PAWN, color):
            if not _is_passed_pawn(board, psq, color):
                continue
            pf, pr = chess.square_file(psq), chess.square_rank(psq)
            nr = pr + fwd
            if not (0 <= nr <= 7):
                continue
            front_sq = chess.square(pf, nr)
            pc = board.piece_at(front_sq)
            if pc and pc.color == enemy and pc.piece_type != chess.PAWN:
                side = "white" if color == chess.WHITE else "black"
                out.append(
                    {
                        "passer": _sq_name(psq),
                        "blocker": _sq_name(front_sq),
                        "side": side,
                    }
                )
    return out


def _opposite_color_bishops(board: chess.Board) -> bool:
    wb = list(board.pieces(chess.BISHOP, chess.WHITE))
    bb = list(board.pieces(chess.BISHOP, chess.BLACK))
    if len(wb) != 1 or len(bb) != 1:
        return False
    wl = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[wb[0]])
    bl = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[bb[0]])
    return wl != bl


def _wrong_rook_pawn_flags(board: chess.Board) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for color in (chess.WHITE, chess.BLACK):
        bishops = list(board.pieces(chess.BISHOP, color))
        if len(bishops) != 1:
            continue
        bsq = bishops[0]
        b_light = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[bsq])
        for psq in board.pieces(chess.PAWN, color):
            f = chess.square_file(psq)
            if f not in (0, 7):
                continue
            if not _is_passed_pawn(board, psq, color):
                continue
            promo_sq = chess.square(f, 7 if color == chess.WHITE else 0)
            promo_light = bool(chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[promo_sq])
            if promo_light != b_light:
                side = "white" if color == chess.WHITE else "black"
                out.append({"side": side, "pawn": _sq_name(psq)})
    return out


def _mop_up_distance(
    board: chess.Board, material: dict[str, Any]
) -> dict[str, Any] | None:
    diff = material.get("diff") if isinstance(material.get("diff"), dict) else {}
    total = diff.get("total")
    if not isinstance(total, (int, float)) or abs(total) < ROUGH_MATERIAL_BALANCE_CP:
        return None
    losing = chess.BLACK if total > 0 else chess.WHITE
    ksq = board.king(losing)
    if ksq is None:
        return None
    corners = (chess.A1, chess.H1, chess.A8, chess.H8)
    dist = min(chess.square_distance(ksq, c) for c in corners)
    side = "black" if losing == chess.BLACK else "white"
    return {"losingSide": side, "kingCornerDist": dist}


def _pawn_heavy_no_major(board: chess.Board) -> bool:
    rq = (
        len(board.pieces(chess.QUEEN, chess.WHITE))
        + len(board.pieces(chess.QUEEN, chess.BLACK))
        + len(board.pieces(chess.ROOK, chess.WHITE))
        + len(board.pieces(chess.ROOK, chess.BLACK))
    )
    return rq == 0


def _rule_of_square_flags(board: chess.Board) -> list[dict[str, Any]]:
    if not _pawn_heavy_no_major(board):
        return []
    out: list[dict[str, Any]] = []
    for color in (chess.WHITE, chess.BLACK):
        enemy_k = board.king(not color)
        promo_r = 7 if color == chess.WHITE else 0
        for psq in board.pieces(chess.PAWN, color):
            if not _is_passed_pawn(board, psq, color):
                continue
            pf, pr = chess.square_file(psq), chess.square_rank(psq)
            promo_sq = chess.square(pf, promo_r)
            steps = (7 - pr) if color == chess.WHITE else pr
            dist_k = chess.square_distance(enemy_k, promo_sq)
            tempo_bonus = 1 if board.turn == color else 0
            wins_race = steps + tempo_bonus <= dist_k
            out.append(
                {
                    "sq": _sq_name(psq),
                    "side": "white" if color == chess.WHITE else "black",
                    "promotionRaceOk": wins_race,
                }
            )
    return out


def _endgame_features(board: chess.Board, material: dict[str, dict]) -> dict[str, Any]:
    """Endgame-only flags (blockades, opposite bishops, wrong rook pawn, mop-up, …)."""
    eg: dict[str, Any] = {}
    blockades = _blockades(board)
    if blockades:
        eg["blockades"] = blockades
    if _opposite_color_bishops(board):
        eg["oppositeColorBishops"] = True
    wrong_rook_pawn = _wrong_rook_pawn_flags(board)
    if wrong_rook_pawn:
        eg["wrongRookPawn"] = wrong_rook_pawn
    mop_up = _mop_up_distance(board, material)
    if mop_up:
        eg["mopUpDistance"] = mop_up
    if _pawn_heavy_no_major(board):
        rule_of_square = _rule_of_square_flags(board)
        if rule_of_square:
            eg["ruleOfTheSquare"] = rule_of_square
    return eg
