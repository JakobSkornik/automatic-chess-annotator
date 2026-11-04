"""Board entities behind a feature or a claim (for hover highlighting in the UI).

``feature_squares(board, name)`` returns the squares a catalog feature is
computed from in ``board``: the pieces it counts, the squares it measures, the
files it reads. Rules that already know their entities (a pin, a capture, a
hole) attach them directly; the facts builder fills the rest from the claimed
features at the claim's support position.
"""
from __future__ import annotations

import chess

from app.models.comment_facts import ClaimEvidence

from .features.pawns import backward as backward_feature
from .features.pawns import isolated as isolated_feature
from .features.pieces._squares import is_outpost_square
from .features.threats import hanging as hanging_feature
from .features.threats import weak_enemies as weak_enemies_feature
from .geometry import _is_passed, _pawn_files, _relative_rank
from .weights import CENTER_SQUARES

_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def piece_entry(board: chess.Board, square: int, role: str = "subject") -> dict | None:
    piece = board.piece_at(square)
    if piece is None:
        return None
    return {"square": chess.square_name(square), "piece": _PIECE_NAMES[piece.piece_type],
            "color": "white" if piece.color else "black", "role": role}


def evidence_for(board: chess.Board, squares, *, roles: dict[int, str] | None = None,
                 baseline_board: chess.Board | None = None, baseline_squares=None,
                 files=None, source: str = "rule") -> ClaimEvidence:
    """Build a ``ClaimEvidence`` from squares in ``board`` (and optionally the start board)."""
    roles = roles or {}
    squares = sorted(set(squares))
    pieces = [entry for sq in squares if (entry := piece_entry(board, sq, roles.get(sq, "subject")))]
    base = sorted(set(baseline_squares if baseline_squares is not None else squares))
    if baseline_board is not None and baseline_squares is None:
        base = [sq for sq in base if baseline_board.piece_at(sq) is not None]
    return ClaimEvidence(squares=[chess.square_name(s) for s in squares], pieces=pieces,
                         baseline_squares=[chess.square_name(s) for s in base],
                         files=sorted(set(files or [])), source=source)


def _side(name: str) -> tuple[chess.Color | None, str]:
    if name.startswith("WHITE_"):
        return chess.WHITE, name[6:]
    if name.startswith("BLACK_"):
        return chess.BLACK, name[6:]
    return None, name


def _king_ring(board: chess.Board, color: chess.Color) -> list[int]:
    king = board.king(color)
    if king is None:
        return []
    return [king, *chess.SquareSet(chess.BB_KING_ATTACKS[king])]


def _pieces(board: chess.Board, color: chess.Color, *types: int) -> list[int]:
    return [sq for t in types for sq in board.pieces(t, color)]


def _file_status(board: chess.Board, square: int, color: chess.Color) -> int:
    file = chess.square_file(square)
    enemy = False
    for sq in board.pieces(chess.PAWN, color):
        if chess.square_file(sq) == file:
            return 0
    for sq in board.pieces(chess.PAWN, not color):
        if chess.square_file(sq) == file:
            enemy = True
    return 1 if enemy else 2


def _bad_bishops(board: chess.Board, color: chess.Color) -> list[int]:
    out = []
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
            out.append(sq)
    return out


def _same_colour_pawns(board: chess.Board, color: chess.Color, bishops) -> list[int]:
    colours = {(chess.square_file(b) + chess.square_rank(b)) % 2 for b in bishops}
    return [p for p in board.pieces(chess.PAWN, color)
            if (chess.square_file(p) + chess.square_rank(p)) % 2 in colours]


def feature_squares(board: chess.Board, name: str) -> tuple[list[int], list[str]]:
    """(squares, files) the feature ``name`` is computed from in ``board``."""
    color, base = _side(name)
    files: list[str] = []
    if color is None:
        if base == "MATERIAL_BALANCE":
            return [], []
        if base == "EVALUATE_PAWNS":
            return [sq for c in (chess.WHITE, chess.BLACK) for sq in _pawn_defects(board, c)], []
        if base == "EVALUATE_KING_SAFETY":
            return [k for c in (chess.WHITE, chess.BLACK) if (k := board.king(c)) is not None], []
        if base == "KING_TROPISM":
            return [k for c in (chess.WHITE, chess.BLACK) if (k := board.king(c)) is not None], []
        if base == "CASTLING_RIGHTS":
            return [sq for c in (chess.WHITE, chess.BLACK) for sq in feature_squares(board, f"{'WHITE' if c else 'BLACK'}_CASTLING_RIGHTS")[0]], []
        return [], []
    enemy = not color
    pawns = list(board.pieces(chess.PAWN, color))
    if base == "MATERIAL":
        return [sq for sq in chess.SquareSet(board.occupied_co[color]) if board.piece_type_at(sq) != chess.KING], []
    if base == "PAWN_DOUBLED":
        doubled = [sqs for f, sqs in _pawn_files(board, color).items() if len(sqs) > 1]
        return [sq for sqs in doubled for sq in sqs], [chess.FILE_NAMES[chess.square_file(sqs[0])] for sqs in doubled]
    if base == "PAWN_ISOLATED":
        return [sq for sq in pawns if not isolated_feature._has_neighbour_on_adjacent_file(board, chess.square_file(sq), color)], []
    if base == "PAWN_BACKWARD":
        return [sq for sq in pawns if backward_feature.is_backward(board, sq, color)], []
    if base == "WEAK_PAWNS":
        iso = feature_squares(board, f"{'WHITE' if color else 'BLACK'}_PAWN_ISOLATED")[0]
        back = feature_squares(board, f"{'WHITE' if color else 'BLACK'}_PAWN_BACKWARD")[0]
        return sorted(set(iso) | set(back)), []
    if base in ("PAWN_PASSED", "OUTSIDE_PASSER"):
        return [sq for sq in pawns if _is_passed(board, sq, color)], []
    if base == "PAWN_DUO":
        return [sq for sq in pawns if any(chess.square_rank(o) == chess.square_rank(sq)
                                          and abs(chess.square_file(o) - chess.square_file(sq)) == 1 for o in pawns)], []
    if base == "PAWN_ADVANCES":
        return [sq for sq in pawns if _relative_rank(sq, color) >= 3], []
    if base == "KNIGHTS_OUTPOSTS":
        return [sq for sq in board.pieces(chess.KNIGHT, color) if is_outpost_square(board, sq, color)], []
    if base == "KNIGHTS_CENTRALIZATION":
        return list(board.pieces(chess.KNIGHT, color)), []
    if base in ("BISHOP_PAIR", "BISHOPS_MOBILITY"):
        return list(board.pieces(chess.BISHOP, color)), []
    if base == "BISHOP_PLUS_PAWNS_ON_COLOR":
        bishops = list(board.pieces(chess.BISHOP, color))
        return bishops + [p for p in _same_colour_pawns(board, color, bishops) if 2 <= chess.square_file(p) <= 5], []
    if base == "BAD_BISHOP":
        bad = _bad_bishops(board, color)
        # The bishop and the central pawns (files c-f) that block it.
        return bad + [p for p in _same_colour_pawns(board, color, bad) if 2 <= chess.square_file(p) <= 5], []
    if base in ("ROOK_OPEN_FILE", "ROOK_HALF_OPEN_FILE"):
        want = 2 if base == "ROOK_OPEN_FILE" else 1
        rooks = [sq for sq in board.pieces(chess.ROOK, color) if _file_status(board, sq, color) == want]
        return rooks, [chess.FILE_NAMES[chess.square_file(sq)] for sq in rooks]
    if base == "ROOK_ON_SEVENTH":
        return [sq for sq in board.pieces(chess.ROOK, color) if _relative_rank(sq, color) == 6], []
    if base == "ROOK_BEHIND_PASSED_PAWN":
        out = []
        passed = [sq for sq in pawns if _is_passed(board, sq, color)]
        for rook in board.pieces(chess.ROOK, color):
            for p in passed:
                if chess.square_file(p) == chess.square_file(rook) and p in board.attacks(rook):
                    behind = chess.square_rank(rook) < chess.square_rank(p) if color else chess.square_rank(rook) > chess.square_rank(p)
                    if behind:
                        out += [rook, p]
        return out, []
    if base == "ROOKS_CONNECTED":
        rooks = list(board.pieces(chess.ROOK, color))
        return rooks if len(rooks) >= 2 and rooks[1] in board.attacks(rooks[0]) else [], []
    if base == "PIECE_ACTIVITY":
        return _pieces(board, color, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN), []
    if base == "CENTER_CONTROL":
        return [c for c in CENTER_SQUARES if board.attackers(color, c)] + \
            sorted({a for c in CENTER_SQUARES for a in board.attackers(color, c)}), []
    if base == "SPACE":
        return _pieces(board, color, chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN), []
    if base == "KING_DANGER":
        ring = _king_ring(board, color)
        attackers = sorted({a for sq in ring for a in board.attackers(enemy, sq) if board.piece_type_at(a) != chess.PAWN})
        return ring + attackers, []
    if base == "KING_ZONE_ATTACKS":
        ring = _king_ring(board, enemy)
        attackers = sorted({a for sq in ring for a in board.attackers(color, sq) if board.piece_type_at(a) != chess.PAWN})
        return ring + attackers, []
    if base == "BACK_RANK":
        king = board.king(color)
        if king is None:
            return [], []
        rank = 0 if color else 7
        return [chess.square(f, rank) for f in range(8)] + [king], []
    if base == "CASTLING_RIGHTS":
        king = board.king(color)
        out = [king] if king is not None else []
        if board.has_kingside_castling_rights(color):
            out.append(chess.H1 if color else chess.H8)
        if board.has_queenside_castling_rights(color):
            out.append(chess.A1 if color else chess.A8)
        return out, []
    if base == "KING_TROPISM":
        king = board.king(enemy)
        return ([king] if king is not None else []) + _pieces(board, color, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN), []
    if base == "WEAK_ENEMIES":
        return [sq for sq in chess.SquareSet(board.occupied_co[enemy]) if weak_enemies_feature.is_weak_enemy(board, sq, color)], []
    if base == "HANGING":
        targets = [sq for sq in chess.SquareSet(board.occupied_co[enemy]) if hanging_feature.is_concretely_hanging(board, sq, color)]
        attackers = sorted({a for t in targets for a in board.attackers(color, t)})
        return targets + attackers, []
    if base == "PINS":
        out = []
        for sq in chess.SquareSet(board.occupied_co[enemy]):
            if board.piece_type_at(sq) != chess.KING and board.is_pinned(enemy, sq):
                out.append(sq)
        king = board.king(enemy)
        return out + ([king] if out and king is not None else []), []
    if base == "KING_ACTIVITY":
        king = board.king(color)
        return [king] if king is not None else [], []
    if base == "PASSER_KING_ESCORT":
        king = board.king(color)
        passed = [sq for sq in pawns if _is_passed(board, sq, color)]
        return ([king] if king is not None and passed else []) + passed, []
    return [], files


def _pawn_defects(board: chess.Board, color: chess.Color) -> list[int]:
    side = "WHITE" if color else "BLACK"
    out: set[int] = set()
    for suffix in ("PAWN_DOUBLED", "PAWN_ISOLATED", "PAWN_BACKWARD", "PAWN_PASSED"):
        out.update(feature_squares(board, f"{side}_{suffix}")[0])
    return sorted(out)


def evidence_from_features(support: chess.Board, baseline: chess.Board | None, features: list[str]) -> ClaimEvidence:
    """Evidence derived from the first two claimed features, at the support node and at the start."""
    squares: set[int] = set()
    files: set[str] = set()
    base: set[int] = set()
    for name in features[:2]:
        sq, fl = feature_squares(support, name)
        squares.update(sq)
        files.update(fl)
        if baseline is not None:
            base.update(feature_squares(baseline, name)[0])
    return evidence_for(support, squares, baseline_squares=base if baseline is not None else None,
                        files=files, source="feature")
