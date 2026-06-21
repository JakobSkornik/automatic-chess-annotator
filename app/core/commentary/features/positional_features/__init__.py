"""Hidden positional features for a board: the per-side feature blocks, board
overlay, endgame pack and final cleanup, assembled by ``compute_hidden_features``.
Split into: util, pawns, pieces_king, endgame, and this assembly module."""

from __future__ import annotations

from typing import Any

import chess

from .endgame import _endgame_features
from .pawns import (
    _backward_pawn_squares,
    _doubled_pawn_squares,
    _isolated_pawn_squares,
    _open_and_semi_open_files,
    _passed_pawn_records,
    _pawn_islands,
    _pawn_structure_classification,
)
from .pieces_king import (
    _attacked_and_attacking,
    _batteries,
    _bishop_quality,
    _center_control,
    _centralization,
    _connectivity,
    _has_connected_rooks,
    _holes,
    _king_exposure,
    _king_pawn_tropism,
    _king_safety,
    _mobility,
    _outposts,
    _rooks_on_files,
    _space,
    _trapped_pieces,
    _weak_squares_full_board,
)
from .util import (
    CASTLING_KEYS,
    CONTESTED_SQUARES_CAP,
    ENDGAME_MINOR_MAJOR_MAX,
    FILE_NAMES,
    KEEP_ZERO_METRICS,
    PROTECTED_TOP_LEVEL_KEYS,
    _is_light_square,
    _minor_major_count,
    _sq_name,
)

__all__ = ["compute_hidden_features"]


def _material_snapshot(board: chess.Board) -> dict[str, dict]:
    values = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
    counts_white = {
        "p": len(board.pieces(chess.PAWN, chess.WHITE)),
        "n": len(board.pieces(chess.KNIGHT, chess.WHITE)),
        "b": len(board.pieces(chess.BISHOP, chess.WHITE)),
        "r": len(board.pieces(chess.ROOK, chess.WHITE)),
        "q": len(board.pieces(chess.QUEEN, chess.WHITE)),
    }
    counts_black = {
        "p": len(board.pieces(chess.PAWN, chess.BLACK)),
        "n": len(board.pieces(chess.KNIGHT, chess.BLACK)),
        "b": len(board.pieces(chess.BISHOP, chess.BLACK)),
        "r": len(board.pieces(chess.ROOK, chess.BLACK)),
        "q": len(board.pieces(chess.QUEEN, chess.BLACK)),
    }
    total_white = sum(counts_white[k] * values[k] for k in values)
    total_black = sum(counts_black[k] * values[k] for k in values)
    diff = {k: counts_white[k] - counts_black[k] for k in values}
    diff["total"] = total_white - total_black

    imbalance: list[str] = []
    if diff["r"] != 0 or diff["b"] != 0 or diff["n"] != 0:
        if diff["r"] == -1 and (diff["b"] + diff["n"]) >= 2:
            imbalance.append("twoMinorsForRook")
        if diff["r"] == 1 and (diff["b"] + diff["n"]) <= -2:
            imbalance.append("rookForTwoMinors")
    if diff["q"] != 0:
        imbalance.append("queenTradeImbalance")

    return {
        "white": {"counts": counts_white, "total": total_white},
        "black": {"counts": counts_black, "total": total_black},
        "diff": diff,
        "imbalance": imbalance,
    }


def _contested_squares(board: chess.Board, cap: int = 10) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for sq in chess.SQUARES:
        wa = len(board.attackers(chess.WHITE, sq))
        ba = len(board.attackers(chess.BLACK, sq))
        if wa > 0 and ba > 0:
            scored.append((wa + ba, {"sq": _sq_name(sq), "white": wa, "black": ba}))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:cap]]


def _opening_tempo_snapshot(board: chess.Board) -> dict[str, int]:
    w_home = sum(
        1
        for pt in (chess.KNIGHT, chess.BISHOP)
        for sq in board.pieces(pt, chess.WHITE)
        if chess.square_rank(sq) == 0
    )
    b_home = sum(
        1
        for pt in (chess.KNIGHT, chess.BISHOP)
        for sq in board.pieces(pt, chess.BLACK)
        if chess.square_rank(sq) == 7
    )
    developed = (4 - min(w_home, 4)) + (4 - min(b_home, 4))
    return {"minorPiecesStillHome": w_home + b_home, "developedScore": developed}


def _is_empty_for_drop(key: str | None, val: Any) -> bool:
    if key in CASTLING_KEYS:
        return False
    if key in KEEP_ZERO_METRICS and isinstance(val, (int, float)) and val == 0:
        return False
    if val is None:
        return True
    if val is False:
        return True
    if isinstance(val, (int, float)) and val == 0:
        return True
    if val == [] or val == {}:
        return True
    return val == ""


def _drop_empty(obj: Any, *, _top_level_key: str | None = None) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if _top_level_key is None and k in PROTECTED_TOP_LEVEL_KEYS:
                out[k] = _drop_empty(v, _top_level_key=k)
                continue
            nv = _drop_empty(v, _top_level_key=k)
            if _is_empty_for_drop(k, nv):
                continue
            out[k] = nv
        return out
    if isinstance(obj, list):
        items = [_drop_empty(x, _top_level_key=_top_level_key) for x in obj]
        items = [x for x in items if not _is_empty_for_drop(None, x)]
        return items
    return obj


def _build_board_overlay(
    files_info: dict[str, list[int]],
    white: dict[str, Any],
    black: dict[str, Any],
    contested: list[dict[str, Any]],
    pawn_structure: dict[str, Any],
    trapped_w: list[dict[str, Any]],
    trapped_b: list[dict[str, Any]],
) -> dict[str, Any]:
    ow = white.get("outposts") if isinstance(white.get("outposts"), dict) else {}
    ob = black.get("outposts") if isinstance(black.get("outposts"), dict) else {}

    def _files_to_letters(idx_list: list[int]) -> list[str]:
        return [FILE_NAMES[i] for i in idx_list]

    overlay: dict[str, Any] = {
        "openFiles": _files_to_letters(list(files_info.get("open", []))),
        "semiOpenWhite": _files_to_letters(list(files_info.get("semiOpenWhite", []))),
        "semiOpenBlack": _files_to_letters(list(files_info.get("semiOpenBlack", []))),
        "weakSquaresWhite": white.get("weakSquares"),
        "weakSquaresBlack": black.get("weakSquares"),
        "holesWhite": white.get("holes"),
        "holesBlack": black.get("holes"),
        "contestedSquares": contested,
        "outpostsWhite": (ow.get("occupied") or []) + (ow.get("available") or []),
        "outpostsBlack": (ob.get("occupied") or []) + (ob.get("available") or []),
        "passedPawnsWhite": [
            x["sq"] for x in white.get("passedPawns", []) if isinstance(x, dict)
        ],
        "passedPawnsBlack": [
            x["sq"] for x in black.get("passedPawns", []) if isinstance(x, dict)
        ],
        "doubledPawnsWhite": white.get("doubledPawns"),
        "doubledPawnsBlack": black.get("doubledPawns"),
        "pawnBreaks": pawn_structure.get("breaks")
        if isinstance(pawn_structure, dict)
        else [],
        "trappedPieces": [
            {"sq": x["sq"], "piece": x["piece"], "color": "w"} for x in trapped_w
        ]
        + [{"sq": x["sq"], "piece": x["piece"], "color": "b"} for x in trapped_b],
    }
    return overlay


def _side_features(
    board: chess.Board,
    color: chess.Color,
    files_info: dict[str, Any],
    is_endgame: bool,
) -> dict[str, Any]:
    """All positional features for one side (absolute, this-side perspective)."""
    side: dict[str, Any] = {}
    side["hasBishopPair"] = len(board.pieces(chess.BISHOP, color)) >= 2
    side["hasKnightPair"] = len(board.pieces(chess.KNIGHT, color)) >= 2
    side["hasRookPair"] = len(board.pieces(chess.ROOK, color)) >= 2
    side["hasQueen"] = len(board.pieces(chess.QUEEN, color)) >= 1

    bishops = list(board.pieces(chess.BISHOP, color))
    side["lightSquareBishops"] = sum(1 for b in bishops if _is_light_square(b))
    side["darkSquareBishops"] = sum(1 for b in bishops if not _is_light_square(b))

    side["canCastleKingSide"] = board.has_kingside_castling_rights(color)
    side["canCastleQueenSide"] = board.has_queenside_castling_rights(color)

    side["doubledPawns"] = _doubled_pawn_squares(board, color)
    side["isolatedPawns"] = _isolated_pawn_squares(board, color)
    side["passedPawns"] = _passed_pawn_records(board, color)
    side["backwardPawns"] = _backward_pawn_squares(board, color)
    side["weakSquares"] = _weak_squares_full_board(board, color)
    side["holes"] = _holes(board, color)
    side["pawnIslands"] = _pawn_islands(board, color)

    side.update(_attacked_and_attacking(board, color))
    side["mobility"] = _mobility(board, color)
    side["centerControl"] = _center_control(board, color)
    side.update(_king_safety(board, color))

    rooks = list(board.pieces(chess.ROOK, color))
    semi_key = "semiOpenWhite" if color == chess.WHITE else "semiOpenBlack"
    side["rooksOnOpenFiles"], side["rooksOnSemiOpenFiles"] = _rooks_on_files(
        rooks, set(files_info.get("open", [])), set(files_info.get(semi_key, []))
    )
    side["connectedRooks"] = _has_connected_rooks(board, rooks)

    side["outposts"] = _outposts(board, color)
    side.update(_bishop_quality(board, color))
    side["space"] = _space(board, color)
    side["centralization"] = _centralization(board, color)
    side["kingExposure"] = _king_exposure(board, color)
    side["batteries"] = _batteries(board, color)

    tropism = _king_pawn_tropism(board, color)
    if tropism is not None:
        side["kingPawnTropism"] = tropism
    side["connectivity"] = _connectivity(board, color)
    side["trappedPieces"] = _trapped_pieces(board, color, skip=is_endgame)
    return side


def _finalize_features(features: dict[str, Any]) -> dict[str, Any]:
    """Drop empty values, but keep the evaluation skeleton sub-objects intact."""
    material = features.pop("material")
    open_files = features.pop("openFiles")
    pawn_structure = features.pop("pawnStructure")
    cleaned = _drop_empty(features)
    cleaned["material"] = material
    cleaned["openFiles"] = open_files
    cleaned["pawnStructure"] = pawn_structure
    return cleaned


def compute_hidden_features(
    board: chess.Board, *, in_opening_book_phase: bool = False
) -> dict[str, Any]:
    """Dense positional features for both sides + UI overlay. Numeric features are absolute per side."""
    files_info = _open_and_semi_open_files(board)
    material = _material_snapshot(board)
    pawn_structure = _pawn_structure_classification(board)
    is_endgame = _minor_major_count(board) < ENDGAME_MINOR_MAJOR_MAX
    contested = _contested_squares(board, cap=CONTESTED_SQUARES_CAP)

    features: dict[str, Any] = {
        "openFiles": files_info,
        "material": material,
        "pawnStructure": pawn_structure,
        "white": _side_features(board, chess.WHITE, files_info, is_endgame),
        "black": _side_features(board, chess.BLACK, files_info, is_endgame),
        "contestedSquares": contested,
    }
    if in_opening_book_phase:
        features["tempo"] = _opening_tempo_snapshot(board)
    if is_endgame:
        endgame = _endgame_features(board, material)
        if endgame:
            features["endgame"] = endgame

    white_side, black_side = features["white"], features["black"]
    features["boardOverlay"] = _build_board_overlay(
        files_info,
        white_side,
        black_side,
        contested,
        pawn_structure,
        white_side.get("trappedPieces") or [],
        black_side.get("trappedPieces") or [],
    )
    return _finalize_features(features)
