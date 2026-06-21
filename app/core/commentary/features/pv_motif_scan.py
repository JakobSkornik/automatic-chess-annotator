"""Scan engine PV lines for tactical/strategic motifs at each ply."""

from __future__ import annotations

from collections.abc import Callable

import chess

from app.core.commentary.features.positional_features import compute_hidden_features
from app.core.commentary.features.strategic_motifs import detect_strategic_motifs
from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
from app.models.chess_events import PlyMotifScan, StrategicMotif, TacticalMotif


def scan_pv_motifs(
    start_board: chess.Board,
    pv_ucis: list[str],
    *,
    max_plies: int = 4,
    phase: str = "middlegame",
    hidden_features_fn: Callable[[chess.Board], dict] = compute_hidden_features,
) -> list[PlyMotifScan]:
    """
    Walk ``pv_ucis`` from ``start_board`` and run motif detectors at each ply.
    Returns one ``PlyMotifScan`` per PV step (1-based ply index within the PV).
    """
    if not pv_ucis:
        return []

    board = start_board.copy()
    out: list[PlyMotifScan] = []
    prev_eval: int | None = None

    for ply_idx, uci in enumerate(pv_ucis[:max_plies], start=1):
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
        except Exception:
            break

        board_before = board.copy()
        san = board.san(move)
        board.push(move)
        board_after = board.copy()

        try:
            hidden_features = hidden_features_fn(board_after)
        except Exception:
            hidden_features = {}

        tact = detect_tactical_motifs(
            board_before,
            board_after,
            move,
            prev_eval,
            None,
        )
        strat = detect_strategic_motifs(
            board_before,
            board_after,
            move,
            hidden_features if isinstance(hidden_features, dict) else {},
            eval_after_cp=None,
            eval_before_cp=prev_eval,
            phase=phase,
            best_pv_ucis=pv_ucis[ply_idx : ply_idx + 2]
            if ply_idx < len(pv_ucis)
            else None,
        )

        out.append(
            PlyMotifScan(
                ply=ply_idx,
                san=san,
                uci=uci,
                mover="white" if board_before.turn == chess.WHITE else "black",
                tactical_motifs=tact,
                strategic_motifs=strat,
            )
        )

    return out


def merge_pv_motifs_into_strategic(
    existing: list[StrategicMotif],
    scans: list[PlyMotifScan],
) -> list[StrategicMotif]:
    """Add unique strategic motifs found along the PV (excluding duplicates)."""
    seen: set[StrategicMotif] = set(existing)
    out = list(existing)
    for s in scans:
        for m in s.strategic_motifs:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def merge_pv_motifs_into_tactical(
    existing: list[TacticalMotif],
    scans: list[PlyMotifScan],
    *,
    exclude_ply_zero: bool = True,
) -> list[TacticalMotif]:
    """Add tactical motifs from PV plies after the root (future threats/plans)."""
    seen: set[TacticalMotif] = set(existing)
    out = list(existing)
    for s in scans:
        if exclude_ply_zero and s.ply <= 1:
            continue
        for m in s.tactical_motifs:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out
