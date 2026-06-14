"""Detect opponent tactical threats from the position after the played move."""

from __future__ import annotations

import chess

from app.core.commentary.features.tactical_motifs import detect_tactical_motifs
from app.models.chess_events import TacticalMotif


def detect_opponent_threats(
    board_after: chess.Board,
    *,
    best_pv_ucis: list[str] | None = None,
    played_matches_best: bool = False,
    max_legal_scan: int = 40,
) -> list[TacticalMotif]:
    """
    Scan what the opponent threatens from ``board_after`` (opponent to move).

    Uses legal-move enumeration plus optional PV[1] from the engine line when the
    played move matched the engine's top choice.
    """
    if board_after.is_game_over():
        return []

    motifs: list[TacticalMotif] = []
    seen: set[TacticalMotif] = set()

    def _add(found: list[TacticalMotif]) -> None:
        for m in found:
            if m not in seen:
                seen.add(m)
                motifs.append(m)

    # Explicit engine PV reply when played == best
    if played_matches_best and best_pv_ucis and len(best_pv_ucis) >= 2:
        try:
            opp_move = chess.Move.from_uci(best_pv_ucis[1])
            if opp_move in board_after.legal_moves:
                ba = board_after.copy()
                ba.push(opp_move)
                _add(
                    detect_tactical_motifs(
                        board_after,
                        ba,
                        opp_move,
                        None,
                        None,
                    )
                )
        except Exception:
            pass

    # Broader scan: all legal opponent moves (capped)
    legal = list(board_after.legal_moves)
    if len(legal) > max_legal_scan:
        legal = legal[:max_legal_scan]

    for opp_move in legal:
        ba = board_after.copy()
        ba.push(opp_move)
        found = detect_tactical_motifs(board_after, ba, opp_move, None, None)
        _add(found)

    return motifs


def threats_prevented(
    opponent_threats: list[TacticalMotif],
    played_motifs: list[TacticalMotif],
) -> list[TacticalMotif]:
    """
    Heuristic: opponent threats that the played move's prophylaxis may address.
    Returns threats not mirrored as immediate tactical themes on the played move.
    """
    played_set = set(played_motifs)
    return [t for t in opponent_threats if t not in played_set]
