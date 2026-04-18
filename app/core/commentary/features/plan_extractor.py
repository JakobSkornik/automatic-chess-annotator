"""Derive plan hints from engine PV lines (destination squares + coarse plan labels)."""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Tuple

import chess


def _uci_seq_from_pv(pv_seq: List[Any]) -> List[str]:
    out: List[str] = []
    for pm in pv_seq or []:
        u = getattr(pm, "move", None)
        if u:
            out.append(str(u))
    return out


def _destination_chain(board: chess.Board, uci_moves: List[str], max_plies: int = 8) -> List[str]:
    """Ordered destination squares along the PV (first line)."""
    b = board.copy()
    out: List[str] = []
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


def _targets_from_uci(board: chess.Board, uci_moves: List[str], min_count: int = 2) -> List[str]:
    b = board.copy()
    dests: List[str] = []
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


def _side_cluster_label(board: chess.Board, dest_square_names: List[str], mover: chess.Color) -> str:
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


def build_plan_comparison(
    fen_before: str,
    pvs: List[List[Any]],
    played_uci: str,
) -> Tuple[List[str], List[str], str, str, List[str], List[str]]:
    """
    Returns (played_targets, best_targets, played_seed, best_seed,
             played_recurring_destinations, best_recurring_destinations).
    Best: PV1 from root. Played: if differs from PV1[0], use PV2 line as proxy for an alternate plan.
    """
    try:
        b0 = chess.Board(fen_before)
    except Exception:
        return [], [], "unclear", "unclear", [], []

    best_ucis = _uci_seq_from_pv(pvs[0]) if pvs and pvs[0] else []
    best_targets = _targets_from_uci(b0, best_ucis)
    best_chain = _destination_chain(b0, best_ucis)
    mover = b0.turn
    best_seed = _side_cluster_label(b0, best_targets, mover)

    played_targets: List[str] = []
    played_seed = "unclear"
    played_chain: List[str] = []
    if best_ucis and played_uci == best_ucis[0]:
        played_targets = best_targets
        played_seed = best_seed
        played_chain = best_chain
    elif len(pvs) > 1 and pvs[1]:
        alt_ucis = _uci_seq_from_pv(pvs[1])
        played_targets = _targets_from_uci(b0, alt_ucis)
        played_seed = _side_cluster_label(b0, played_targets, mover)
        played_chain = _destination_chain(b0, alt_ucis)
    elif best_ucis:
        played_seed = "deviation_from_engine_top_line"

    return played_targets, best_targets, played_seed, best_seed, played_chain, best_chain
