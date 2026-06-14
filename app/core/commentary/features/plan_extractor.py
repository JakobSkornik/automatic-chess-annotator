"""Derive plan hints from engine PV lines (destination squares + coarse plan labels)."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any

import chess


class PlanTag(str, Enum):
    PAWN_STORM = "pawn_storm"
    PIECE_MANEUVER = "piece_maneuver"
    ROOK_INVASION = "rook_invasion"
    PAWN_BREAK = "pawn_break"
    KING_ACTIVATION = "king_activation"
    CENTRALIZATION = "centralization"
    EXCHANGE = "exchange"


def _uci_seq_from_pv(pv_seq: list[Any]) -> list[str]:
    out: list[str] = []
    for pm in pv_seq or []:
        u = getattr(pm, "move", None)
        if u:
            out.append(str(u))
    return out


def _destination_chain(
    board: chess.Board, uci_moves: list[str], max_plies: int = 8
) -> list[str]:
    """Ordered destination squares along the PV (first line)."""
    b = board.copy()
    out: list[str] = []
    for u in uci_moves[:max_plies]:
        try:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                break
            out.append(chess.square_name(m.to_square))
            b.push(m)
        except Exception:
            break
    return out


def _targets_from_uci(
    board: chess.Board, uci_moves: list[str], min_count: int = 2
) -> list[str]:
    b = board.copy()
    dests: list[str] = []
    for u in uci_moves[:12]:
        try:
            m = chess.Move.from_uci(u)
            if m not in b.legal_moves:
                break
            dests.append(chess.square_name(m.to_square))
            b.push(m)
        except Exception:
            break
    c = Counter(dests)
    return [sq for sq, n in c.most_common(8) if n >= min_count]


def _side_cluster_label(
    board: chess.Board, dest_square_names: list[str], mover: chess.Color
) -> str:
    if not dest_square_names:
        return "unclear"
    files = [ord(s[0]) - ord("a") for s in dest_square_names if len(s) >= 2]
    if not files:
        return "unclear"
    avg = sum(files) / len(files)
    if mover == chess.WHITE:
        if avg >= 5:
            return "kingside_activity"
        if avg <= 2:
            return "queenside_activity"
    else:
        if avg <= 2:
            return "kingside_activity"
        if avg >= 5:
            return "queenside_activity"
    return "central_play"


def classify_pv_plan(
    board: chess.Board,
    uci_moves: list[str],
    *,
    max_plies: int = 8,
) -> list[str]:
    """Classify PV move sequence into named plan tags."""
    tags: list[str] = []
    b = board.copy()
    pawn_advances_wing = 0
    piece_repositions = 0
    rook_to_seventh = False
    rook_to_open_file = False
    king_walk = 0
    captures = 0
    central_piece_moves = 0
    dest_files: list[int] = []

    for uci in uci_moves[:max_plies]:
        try:
            m = chess.Move.from_uci(uci)
            if m not in b.legal_moves:
                break
        except Exception:
            break

        piece = b.piece_at(m.from_square)
        if piece is None:
            break

        mover = piece.color
        fr, ff = chess.square_rank(m.from_square), chess.square_file(m.from_square)
        tr, tf = chess.square_rank(m.to_square), chess.square_file(m.to_square)
        dest_files.append(tf)

        if piece.piece_type == chess.PAWN:
            if (
                ff == tf
                and (
                    (mover == chess.WHITE and tr > fr)
                    or (mover == chess.BLACK and tr < fr)
                )
                and (tf >= 5 or tf <= 2)
            ):
                pawn_advances_wing += 1
            if b.is_capture(m):
                tags.append(PlanTag.PAWN_BREAK.value)
        elif piece.piece_type == chess.ROOK:
            if (mover == chess.WHITE and tr == 6) or (mover == chess.BLACK and tr == 1):
                rook_to_seventh = True
            f = tf
            wpf = any(
                chess.square(f, r) in b.pieces(chess.PAWN, chess.WHITE)
                for r in range(8)
            )
            bpf = any(
                chess.square(f, r) in b.pieces(chess.PAWN, chess.BLACK)
                for r in range(8)
            )
            if not wpf and not bpf:
                rook_to_open_file = True
        elif piece.piece_type == chess.KING:
            if 2 <= tf <= 5 and 2 <= tr <= 5:
                king_walk += 1
        else:
            if 2 <= tf <= 5 and 2 <= tr <= 5:
                central_piece_moves += 1
            if abs(tf - ff) + abs(tr - fr) >= 2:
                piece_repositions += 1

        if b.is_capture(m):
            captures += 1

        b.push(m)

    if pawn_advances_wing >= 2:
        tags.append(PlanTag.PAWN_STORM.value)
    if piece_repositions >= 2 or central_piece_moves >= 2:
        tags.append(PlanTag.PIECE_MANEUVER.value)
    if rook_to_seventh or (rook_to_open_file and piece_repositions >= 1):
        tags.append(PlanTag.ROOK_INVASION.value)
    if king_walk >= 2:
        tags.append(PlanTag.KING_ACTIVATION.value)
    if central_piece_moves >= 2 and pawn_advances_wing == 0:
        tags.append(PlanTag.CENTRALIZATION.value)
    if captures >= 2:
        tags.append(PlanTag.EXCHANGE.value)

    # Repeated destination region
    if len(dest_files) >= 3:
        c = Counter(dest_files)
        if c.most_common(1)[0][1] >= 2 and not tags:
            tags.append(PlanTag.PIECE_MANEUVER.value)

    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_plan_comparison(
    fen_before: str,
    pvs: list[list[Any]],
    played_uci: str,
) -> tuple[list[str], list[str], str, str, list[str], list[str], list[str], list[str]]:
    """
    Returns (played_targets, best_targets, played_seed, best_seed,
             played_recurring_destinations, best_recurring_destinations,
             played_plan_tags, best_plan_tags).
    Best: PV1 from root. Played: if differs from PV1[0], use PV2 line as proxy for an alternate plan.
    """
    try:
        b0 = chess.Board(fen_before)
    except Exception:
        return [], [], "unclear", "unclear", [], [], [], []

    best_ucis = _uci_seq_from_pv(pvs[0]) if pvs and pvs[0] else []
    best_targets = _targets_from_uci(b0, best_ucis)
    best_chain = _destination_chain(b0, best_ucis)
    best_tags = classify_pv_plan(b0, best_ucis)
    mover = b0.turn
    best_seed = _side_cluster_label(b0, best_targets, mover)

    played_targets: list[str] = []
    played_seed = "unclear"
    played_chain: list[str] = []
    played_tags: list[str] = []
    if best_ucis and played_uci == best_ucis[0]:
        played_targets = best_targets
        played_seed = best_seed
        played_chain = best_chain
        played_tags = best_tags
    elif len(pvs) > 1 and pvs[1]:
        alt_ucis = _uci_seq_from_pv(pvs[1])
        played_targets = _targets_from_uci(b0, alt_ucis)
        played_seed = _side_cluster_label(b0, played_targets, mover)
        played_chain = _destination_chain(b0, alt_ucis)
        played_tags = classify_pv_plan(b0, alt_ucis)
    elif best_ucis:
        played_seed = "deviation_from_engine_top_line"

    return (
        played_targets,
        best_targets,
        played_seed,
        best_seed,
        played_chain,
        best_chain,
        played_tags,
        best_tags,
    )
