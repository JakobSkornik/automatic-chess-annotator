"""Heuristic strategic and endgame motif tags from hidden features + board."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import chess

from app.models.chess_events import StrategicMotif


def _backward_enemy_pawn_exists(board: chess.Board, mover: chess.Color) -> bool:
    """Rough backward-pawn tag on the opponent (Carlsbad-style targets)."""
    enemy = not mover
    pf = _pawn_files(board, enemy)
    for f, lst in pf.items():
        if len(lst) > 1:
            continue
        sq = lst[0]
        r = chess.square_rank(sq)
        has_l = (f - 1) in pf
        has_r = (f + 1) in pf
        if enemy == chess.WHITE:
            blocked = r > 0 and board.piece_at(chess.square(f, r - 1)) == chess.Piece(
                chess.PAWN, chess.BLACK
            )
        else:
            blocked = r < 7 and board.piece_at(chess.square(f, r + 1)) == chess.Piece(
                chess.PAWN, chess.WHITE
            )
        if blocked and not has_l and not has_r:
            return True
    return False


def _pawn_files(board: chess.Board, color: chess.Color) -> Dict[int, List[int]]:
    files: Dict[int, List[int]] = {}
    for sq in board.pieces(chess.PAWN, color):
        fi = chess.square_file(sq)
        files.setdefault(fi, []).append(sq)
    return files

MATE_SCORE = 1_000_000


def _dedupe(xs: List[StrategicMotif]) -> List[StrategicMotif]:
    seen: Set[StrategicMotif] = set()
    out: List[StrategicMotif] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def detect_strategic_motifs(
    board_before: chess.Board,
    board_after: chess.Board,
    move: chess.Move,
    hidden_features: Optional[Dict[str, Any]],
    *,
    eval_after_cp: Optional[int] = None,
    phase: str = "middlegame",
    best_pv_ucis: Optional[List[str]] = None,
) -> List[StrategicMotif]:
    """Return strategic/endgame motif tags (best-effort heuristics)."""
    hf = hidden_features if isinstance(hidden_features, dict) else {}
    motifs: List[StrategicMotif] = []

    piece_after = board_after.piece_at(move.to_square)
    if piece_after is None:
        return []
    mover = piece_after.color
    w = hf.get("white") if isinstance(hf.get("white"), dict) else {}
    b = hf.get("black") if isinstance(hf.get("black"), dict) else {}
    ps = hf.get("pawnStructure") if isinstance(hf.get("pawnStructure"), dict) else {}
    mat = hf.get("material") if isinstance(hf.get("material"), dict) else {}

    # --- From hidden feature extractors (positional_features) ---
    out_w = w.get("outposts") if isinstance(w.get("outposts"), dict) else {}
    out_b = b.get("outposts") if isinstance(b.get("outposts"), dict) else {}
    occ = (out_w.get("occupied") or []) + (out_b.get("occupied") or [])
    avail = (out_w.get("available") or []) + (out_b.get("available") or [])
    if occ or avail:
        motifs.append(StrategicMotif.OUTPOST)

    if isinstance(w.get("badBishops"), int) and w["badBishops"] >= 1:
        motifs.append(StrategicMotif.BAD_BISHOP)
    if isinstance(b.get("badBishops"), int) and b["badBishops"] >= 1:
        motifs.append(StrategicMotif.BAD_BISHOP)

    weak_w = w.get("weakSquares") if isinstance(w.get("weakSquares"), list) else []
    weak_b = b.get("weakSquares") if isinstance(b.get("weakSquares"), list) else []
    if (mover == chess.WHITE and weak_b) or (mover == chess.BLACK and weak_w):
        motifs.append(StrategicMotif.WEAK_SQUARE_CREATION)

    if isinstance(w.get("lightSquareBishops"), int) and isinstance(
        w.get("darkSquareBishops"), int
    ):
        if abs(w["lightSquareBishops"] - w.get("darkSquareBishops", 0)) >= 2:
            motifs.append(StrategicMotif.COLOR_COMPLEX_WEAKNESS)

    # IQP / hanging pawns (heuristic from counts + center type)
    ct = ps.get("centerType") if isinstance(ps, dict) else None
    if isinstance(w.get("isolatedPawns"), int) and w["isolatedPawns"] >= 1 and ct in (
        "open",
        "semi-open",
        "small",
    ):
        motifs.append(StrategicMotif.ISOLATED_QUEEN_PAWN)
    if isinstance(w.get("doubledPawns"), int) and w["doubledPawns"] >= 1:
        if isinstance(b.get("doubledPawns"), int) and b["doubledPawns"] >= 1:
            motifs.append(StrategicMotif.HANGING_PAWNS)

    # Open file: rook/queen on open file
    for sq in board_after.pieces(chess.ROOK, mover) | board_after.pieces(chess.QUEEN, mover):
        f = chess.square_file(sq)
        wpf = any(
            chess.square(f, r) in board_after.pieces(chess.PAWN, chess.WHITE) for r in range(8)
        )
        bpf = any(
            chess.square(f, r) in board_after.pieces(chess.PAWN, chess.BLACK) for r in range(8)
        )
        if not wpf and not bpf:
            motifs.append(StrategicMotif.OPEN_FILE_OCCUPATION)
            break

    # Minority attack pattern: a-b-c pawns vs d-e for same side (very rough)
    if ct == "semi-open" and isinstance(ps.get("tension"), list) and len(ps["tension"]) >= 2:
        motifs.append(StrategicMotif.MINORITY_ATTACK)

    # Castling opposite sides
    wk = board_after.king(chess.WHITE)
    bk = board_after.king(chess.BLACK)
    if wk is not None and bk is not None:
        wf, wr = chess.square_file(wk), chess.square_rank(wk)
        bf, br = chess.square_file(bk), chess.square_rank(bk)
        if (wr <= 2 and br >= 6) or (wr >= 6 and br <= 2):
            motifs.append(StrategicMotif.OPPOSITE_SIDE_CASTLING_RACE)

    if _backward_enemy_pawn_exists(board_after, mover):
        motifs.append(StrategicMotif.BACKWARD_PAWN_TARGET)

    # Prophylaxis: quiet move that attacks the landing square of opponent's 2nd PV move (proxy)
    pv_ucis = best_pv_ucis or []
    if (
        len(pv_ucis) >= 2
        and not board_before.is_capture(move)
        and not board_after.is_check()
    ):
        try:
            opp = chess.Move.from_uci(pv_ucis[1])
            if opp.to_square in board_after.attacks(move.to_square):
                motifs.append(StrategicMotif.PROPHYLAXIS)
        except Exception:
            pass

    pa = board_after.piece_at(move.to_square)
    if pa and pa.piece_type == chess.KNIGHT:
        fr, tr = chess.square_rank(move.from_square), chess.square_rank(move.to_square)
        ff, tf = chess.square_file(move.from_square), chess.square_file(move.to_square)
        rim = ff in (0, 7) or fr in (0, 7)
        central = 2 <= tf <= 5 and 2 <= tr <= 5
        if rim and central:
            motifs.append(StrategicMotif.PIECE_REROUTING)

    # Domination: minor piece attacks enemy minor / rook with few escapes (very rough)
    enemy = not mover
    for sq in chess.SQUARES:
        p = board_after.piece_at(sq)
        if not p or p.color != enemy:
            continue
        if p.piece_type not in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
            continue
        if sq not in board_after.attacks(move.to_square):
            continue
        legal_escapes = 0
        for m in board_after.legal_moves:
            if m.from_square == sq:
                legal_escapes += 1
        if legal_escapes <= 2 and p.piece_type != chess.KING:
            motifs.append(StrategicMotif.DOMINATION)
            break

    # Endgame motifs
    if phase == "endgame":
        minors = (
            len(board_after.pieces(chess.KNIGHT, chess.WHITE))
            + len(board_after.pieces(chess.BISHOP, chess.WHITE))
            + len(board_after.pieces(chess.KNIGHT, chess.BLACK))
            + len(board_after.pieces(chess.BISHOP, chess.BLACK))
        )
        rq = (
            len(board_after.pieces(chess.ROOK, chess.WHITE))
            + len(board_after.pieces(chess.QUEEN, chess.WHITE))
            + len(board_after.pieces(chess.ROOK, chess.BLACK))
            + len(board_after.pieces(chess.QUEEN, chess.BLACK))
        )
        if minors + rq <= 2 and wk and bk:
            # crude opposition: kings on same color, one file apart
            if (chess.square_file(wk) - chess.square_file(bk)) % 2 == 0:
                if abs(chess.square_rank(wk) - chess.square_rank(bk)) <= 2:
                    motifs.append(StrategicMotif.OPPOSITION)

        for c in (chess.WHITE, chess.BLACK):
            for sq in board_after.pieces(chess.PAWN, c):
                if _is_outside_passer(board_after, sq, c):
                    motifs.append(StrategicMotif.OUTSIDE_PASSER)
                    break

        if eval_after_cp is not None and abs(eval_after_cp) < MATE_SCORE - 500:
            if board_after.legal_moves.count() <= 3 and rq == 0:
                motifs.append(StrategicMotif.ZUGZWANG)

        # KRPKR: crude Lucena / Philidor labels
        w_rooks = len(board_after.pieces(chess.ROOK, chess.WHITE))
        b_rooks = len(board_after.pieces(chess.ROOK, chess.BLACK))
        w_pawns = len(board_after.pieces(chess.PAWN, chess.WHITE))
        b_pawns = len(board_after.pieces(chess.PAWN, chess.BLACK))
        if w_rooks == 1 and b_rooks == 1 and w_pawns + b_pawns == 1 and minors + rq <= 2:
            pr = next(iter(board_after.pieces(chess.PAWN, chess.WHITE) or []), None) or next(
                iter(board_after.pieces(chess.PAWN, chess.BLACK) or []), None
            )
            if pr is not None:
                pr_rank = chess.square_rank(pr)
                if mover == chess.WHITE and pr_rank >= 5:
                    motifs.append(StrategicMotif.LUCENA)
                elif mover == chess.BLACK and pr_rank <= 2:
                    motifs.append(StrategicMotif.LUCENA)
                if mover == chess.WHITE and pr_rank <= 4:
                    motifs.append(StrategicMotif.PHILIDOR)
                elif mover == chess.BLACK and pr_rank >= 3:
                    motifs.append(StrategicMotif.PHILIDOR)

        if rq == 0 and minors == 0 and w_pawns == 0 and b_pawns == 0:
            w_maj = len(board_after.pieces(chess.ROOK, chess.WHITE)) + len(
                board_after.pieces(chess.QUEEN, chess.WHITE)
            )
            b_maj = len(board_after.pieces(chess.ROOK, chess.BLACK)) + len(
                board_after.pieces(chess.QUEEN, chess.BLACK)
            )
            if w_maj <= 1 and b_maj <= 1 and eval_after_cp is not None:
                if abs(eval_after_cp) < 80 and board_after.legal_moves.count() <= 8:
                    motifs.append(StrategicMotif.FORTRESS)

        # Triangulation: king moved to same-colored square twice in 3 king moves — omitted (needs history)

    return _dedupe(motifs)


def _is_outside_passer(board: chess.Board, pawn_sq: int, color: chess.Color) -> bool:
    """Passed pawn on a-file or h-file heuristic."""
    f = chess.square_file(pawn_sq)
    if f not in (0, 7):
        return False
    enemy = not color
    pr = chess.square_rank(pawn_sq)
    if color == chess.WHITE:
        ranks = range(pr + 1, 8)
    else:
        ranks = range(pr - 1, -1, -1)
    for r in ranks:
        for df in (-1, 0, 1):
            nf = f + df
            if 0 <= nf <= 7:
                sq = chess.square(nf, r)
                if sq in board.pieces(chess.PAWN, enemy):
                    return False
    return True
