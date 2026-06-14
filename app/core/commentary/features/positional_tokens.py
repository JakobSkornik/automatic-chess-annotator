"""Bolčič 2024-style lexical tokens for BM25 positional similarity (static + dynamic fields)."""

from __future__ import annotations

from typing import Any

import chess

ENCODER_VERSION = "2"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _chebyshev(a: int, b: int) -> int:
    af, ar = chess.square_file(a), chess.square_rank(a)
    bf, br = chess.square_file(b), chess.square_rank(b)
    return max(abs(af - bf), abs(ar - br))


def _naive_piece_tokens(board: chess.Board) -> list[str]:
    """Piece type + square from a1..h8 (white's view), paper §4.2.1."""
    out: list[str] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            out.append(f"{p.symbol()}{chess.square_name(sq)}")
    return out


def _reachable_weighted_tokens(board: chess.Board) -> list[str]:
    """Piece + target square + Chebyshev distance weight, paper §2.2.2."""
    out: list[str] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        sym = p.symbol()
        for atk_sq in board.attacks(sq):
            d = _chebyshev(sq, atk_sq)
            w = 1.0 - 7 * d / 64
            w = max(0.125, round(w, 2))
            out.append(f"{sym}{chess.square_name(atk_sq)}|{w:.2f}")
    return out


def _attack_defense_tokens(board: chess.Board) -> tuple[list[str], list[str]]:
    """Attacks and defenses between pieces, paper §2.2.3."""
    attacks: list[str] = []
    defenses: list[str] = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        sym = p.symbol()
        atk_mask = board.attacks(sq)
        for tgt in chess.SQUARES:
            if tgt == sq:
                continue
            if not (atk_mask & chess.BB_SQUARES[tgt]):
                continue
            victim = board.piece_at(tgt)
            if not victim:
                continue
            if victim.color != p.color:
                attacks.append(f"{sym}>{victim.symbol()}{chess.square_name(tgt)}")
            else:
                defenses.append(f"{sym}<{victim.symbol()}{chess.square_name(tgt)}")
    return attacks, defenses


def _pawn_files(board: chess.Board, color: chess.Color) -> dict[int, list[int]]:
    files: dict[int, list[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        fi = chess.square_file(sq)
        files.setdefault(fi, []).append(sq)
    return files


def _is_passed(board: chess.Board, pawn_sq: int, color: chess.Color) -> bool:
    pawn_file = chess.square_file(pawn_sq)
    enemy = not color
    if color == chess.WHITE:
        ranks = range(chess.square_rank(pawn_sq) + 1, 8)
    else:
        ranks = range(chess.square_rank(pawn_sq) - 1, -1, -1)
    for rank in ranks:
        for f in (pawn_file - 1, pawn_file, pawn_file + 1):
            if 0 <= f <= 7:
                sq = chess.square(f, rank)
                if sq in board.pieces(chess.PAWN, enemy):
                    return False
    return True


def _pawn_structure_tokens(board: chess.Board) -> list[str]:
    """Pawn structure tags (paper Table 2.2, simplified)."""
    tags: list[str] = []
    for color in (chess.WHITE, chess.BLACK):
        prefix = "w" if color == chess.WHITE else "b"
        pf = _pawn_files(board, color)
        for f, lst in pf.items():
            if len(lst) > 1:
                tags.append(f"doubled_{prefix}{chess.FILE_NAMES[f]}")
        for f, lst in pf.items():
            has_l = (f - 1) in pf
            has_r = (f + 1) in pf
            if not has_l and not has_r:
                for sq in lst:
                    tags.append(f"isolated_{prefix}{chess.square_name(sq)}")
        for sq in board.pieces(chess.PAWN, color):
            if _is_passed(board, sq, color):
                tags.append(f"passed_{prefix}{chess.square_name(sq)}")
        for f, lst in pf.items():
            if len(lst) > 1:
                continue
            sq = lst[0]
            r = chess.square_rank(sq)
            has_l = (f - 1) in pf
            has_r = (f + 1) in pf
            if color == chess.WHITE:
                blocked = r > 0 and board.piece_at(
                    chess.square(f, r - 1)
                ) == chess.Piece(chess.PAWN, chess.BLACK)
            else:
                blocked = r < 7 and board.piece_at(
                    chess.square(f, r + 1)
                ) == chess.Piece(chess.PAWN, chess.WHITE)
            if blocked and not has_l and not has_r:
                tags.append(f"backward_{prefix}{chess.square_name(sq)}")
    return tags[:80]


def _center_type_token(board: chess.Board) -> str:
    """Single center type token (paper §2.2.5, simplified)."""
    wps = board.pieces(chess.PAWN, chess.WHITE)
    bps = board.pieces(chess.PAWN, chess.BLACK)
    center = {chess.D4, chess.D5, chess.E4, chess.E5}
    wp_c = [sq for sq in wps if sq in center]
    bp_c = [sq for sq in bps if sq in center]
    if not wp_c and not bp_c:
        # any piece in center?
        occ = sum(1 for sq in center if board.piece_at(sq))
        if occ == 0:
            return "open"
    if len(wp_c) >= 1 and len(bp_c) >= 1:
        # locked / blocked heuristic
        for wf in range(8):
            for wr in range(7):
                if chess.square(wf, wr) in wps and chess.square(wf, wr + 1) in bps:
                    return "closed"
    if len(wp_c) == 2 and chess.D4 in wp_c and chess.E4 in wp_c:
        return "classical_mobile"
    if (len(wp_c) + len(bp_c)) <= 2 and (wp_c or bp_c):
        return "small"
    if wp_c and bp_c:
        return "blocked"
    return "tense"


def king_placement_signature(board: chess.Board) -> str:
    """Lexical tag for king files / castling side (BM25 field)."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return ""
    wf, wr = chess.square_file(wk), chess.square_rank(wk)
    bf, br = chess.square_file(bk), chess.square_rank(bk)
    w_castle = "qs" if wf <= 3 else ("ks" if wf >= 6 else "mid")
    b_castle = "qs" if bf <= 3 else ("ks" if bf >= 6 else "mid")
    race = "opposite_side" if abs(wr - br) >= 4 else "same_side"
    return f"wK{chess.square_name(wk)} bK{chess.square_name(bk)} w_{w_castle} b_{b_castle} {race}"


def imbalance_signature(board: chess.Board) -> str:
    """Material / structure imbalance tags for retrieval."""
    parts: list[str] = []
    for c, tag in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        if len(board.pieces(chess.BISHOP, c)) >= 2:
            parts.append(f"bishop_pair_{tag}")
    wq = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if (
        wq == 0
        and len(board.pieces(chess.PAWN, chess.WHITE))
        + len(board.pieces(chess.PAWN, chess.BLACK))
        <= 6
    ):
        parts.append("queenless_endgame")
    return " ".join(parts)


def _material_value(board: chess.Board, color: chess.Color) -> int:
    s = 0
    for pt in chess.PIECE_TYPES:
        if pt == chess.KING:
            continue
        for _ in board.pieces(pt, color):
            s += PIECE_VALUES.get(pt, 0)
    return s


def _dynamic_general_from_pv(board: chess.Board, pv_san: list[str]) -> list[str]:
    """High-level events from PV (captures, checks, promotions)."""
    tags: list[str] = []
    b = board.copy()
    for san in pv_san[:12]:
        if not san:
            continue
        try:
            move = b.parse_san(san)
        except Exception:
            break
        if b.is_capture(move):
            tags.append("capture")
        if b.gives_check(move):
            tags.append("check")
        promotion = move.promotion
        if promotion:
            tags.append("promotion")
        b.push(move)
    # sacrifice: material loss vs baseline
    # simplified: any capture sequence where side loses more material in window
    return list(dict.fromkeys(tags))  # dedupe preserve order


def _dynamic_solution_tokens(board: chess.Board, pv_san: list[str]) -> list[str]:
    """SAN sequence with $ prefix; first 5 plies; paper §4.2.2 / §7.1."""
    out: list[str] = []
    b = board.copy()
    for san in pv_san[:5]:
        if not san:
            continue
        out.append(f"${san}")
        try:
            move = b.parse_san(san)
            if b.is_capture(move):
                cap = b.piece_at(move.to_square)
                sym = cap.symbol() if cap else "x"
                out.append(f"#x{sym}")
            b.push(move)
        except Exception:
            break
    return out


def encode_position(board: chess.Board, pv_san: list[str]) -> dict[str, Any]:
    """
    Return the five BM25 text fields plus metadata for Tantivy indexing.

    Fields:
      static_attributes, pawn_structure, center, dynamic_general, dynamic_solution
    """
    naive = _naive_piece_tokens(board)
    reach = _reachable_weighted_tokens(board)
    atk, defe = _attack_defense_tokens(board)
    static_parts = naive + reach + atk + defe
    static_attributes = " ".join(static_parts)

    pawn_structure = " ".join(_pawn_structure_tokens(board))
    center = _center_type_token(board)

    dg = _dynamic_general_from_pv(board, pv_san)
    dynamic_general = " ".join(dg)

    ds = _dynamic_solution_tokens(board, pv_san)
    dynamic_solution = " ".join(ds)

    return {
        "static_attributes": static_attributes,
        "pawn_structure": pawn_structure,
        "center": center,
        "dynamic_general": dynamic_general,
        "dynamic_solution": dynamic_solution,
        "king_placement": king_placement_signature(board),
        "imbalance_signature": imbalance_signature(board),
        "player_color": "w" if board.turn == chess.WHITE else "b",
        "fen": board.fen(),
        "pv_san": " ".join(pv_san[:8]),
        "encoder_version": ENCODER_VERSION,
    }


def encode_position_strings_only(
    board: chess.Board, pv_san: list[str]
) -> dict[str, str]:
    """Same as encode_position but only the five indexed string fields."""
    d = encode_position(board, pv_san)
    return {
        "static_attributes": d["static_attributes"],
        "pawn_structure": d["pawn_structure"],
        "center": d["center"],
        "dynamic_general": d["dynamic_general"],
        "dynamic_solution": d["dynamic_solution"],
        "king_placement": d.get("king_placement", ""),
        "imbalance_signature": d.get("imbalance_signature", ""),
    }
