"""Compare played engine line vs best engine line N plies deep (future positions)."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import chess

from app.core.engine.engine_connector import EngineConnector
from app.core.commentary.features.positional_features import compute_hidden_features
from app.models.chess_events import FutureLineDelta

logger = logging.getLogger(__name__)

MATE_SCORE = 1_000_000


def _info_to_white_cp(info: Any) -> Optional[int]:
    if not isinstance(info, dict):
        return None
    sc = info.get("score")
    if sc is None:
        return None
    try:
        return int(sc.white().score(mate_score=MATE_SCORE))
    except Exception:
        return None


def _normalize_info(info: Any) -> Optional[dict]:
    if isinstance(info, list) and info:
        info = info[0]
    return info if isinstance(info, dict) else None


def _pv_moves_from_info(
    engine: EngineConnector, board: chess.Board, depth: int, max_plies: int
) -> Tuple[List[chess.Move], Optional[int]]:
    """Return PV line as chess.Move list and root eval (White POV cp)."""
    info = _normalize_info(engine.analyse(board, depth=depth, multiPv=1))
    if not info:
        return [], None
    root_cp = _info_to_white_cp(info)
    pv = info.get("pv") or []
    moves: List[chess.Move] = []
    b = board.copy()
    for m in pv[:max_plies]:
        if m not in b.legal_moves:
            break
        moves.append(m)
        b.push(m)
    return moves, root_cp


def _leaf_eval_after_line(
    engine: EngineConnector, start: chess.Board, moves: List[chess.Move], depth: int
) -> Optional[int]:
    """Evaluate position after playing `moves` from `start` (White POV cp)."""
    b = start.copy()
    for m in moves:
        if m not in b.legal_moves:
            return None
        b.push(m)
    info = _normalize_info(engine.analyse(b, depth=depth, multiPv=1))
    return _info_to_white_cp(info) if info else None


def _flatten_numeric_features(feat: Dict[str, Any]) -> Dict[str, float]:
    """Extract comparable numeric scalars for delta (both sides + pawn center)."""
    out: Dict[str, float] = {}

    def _coerce_numeric(side: Dict[str, Any], k: str) -> Optional[float]:
        v = side.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list):
            return float(len(v))
        return None

    for side in ("white", "black"):
        d = feat.get(side)
        if not isinstance(d, dict):
            continue
        for k in (
            "mobility",
            "centerControl",
            "kingExposure",
            "kingShieldPawns",
            "kingZoneAttacks",
            "passedPawns",
            "isolatedPawns",
            "doubledPawns",
            "backwardPawns",
            "attackingPieces",
            "attackedPieces",
            "centralization",
            "space",
        ):
            num = _coerce_numeric(d, k)
            if num is not None:
                out[f"{side}.{k}"] = num
    return out


def feature_deltas_between_fens(fen_a: str, fen_b: str, limit: int = 6) -> Dict[str, float]:
    """Public helper for tests: compare hidden features at two positions."""
    try:
        fa = compute_hidden_features(chess.Board(fen_a))
        fb = compute_hidden_features(chess.Board(fen_b))
    except Exception:
        return {}
    return _top_feature_deltas(fa, fb, limit=limit)


def _top_feature_deltas(
    played_feat: Dict[str, Any], best_feat: Dict[str, Any], limit: int = 6
) -> Dict[str, float]:
    a = _flatten_numeric_features(played_feat)
    b = _flatten_numeric_features(best_feat)
    keys = set(a) | set(b)
    deltas: List[Tuple[str, float]] = []
    for k in keys:
        va = a.get(k, 0.0)
        vb = b.get(k, 0.0)
        deltas.append((k, vb - va))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    return {k: round(v, 3) for k, v in deltas[:limit]}


def _destination_targets(moves: List[chess.Move], min_count: int = 2) -> List[str]:
    dests = [chess.square_name(m.to_square) for m in moves]
    c = Counter(dests)
    return [sq for sq, n in c.most_common(8) if n >= min_count]


def compare_played_vs_best_future_lines(
    engine: EngineConnector,
    fen_before: str,
    fen_after_played: str,
    best_move_uci: Optional[str],
    played_move_uci: str,
    *,
    depth: int = 18,
    n_plies: int = 6,
) -> Optional[FutureLineDelta]:
    """
    From positions after the played move and after the engine best first move,
    run multipv=1 PVs and compare leaf features after n_plies plies along each line.
    """
    if not best_move_uci or best_move_uci == played_move_uci:
        return None
    try:
        bb = chess.Board(fen_before)
        bm = chess.Move.from_uci(best_move_uci)
        if bm not in bb.legal_moves:
            return None
        bb.push(bm)
        fen_after_best = bb.fen()
    except Exception as e:
        logger.debug("future_line_compare: bad best move: %s", e)
        return None

    try:
        b_played = chess.Board(fen_after_played)
        b_best = chess.Board(fen_after_best)
    except Exception:
        return None

    played_moves, _ = _pv_moves_from_info(engine, b_played, depth, n_plies)
    best_moves, _ = _pv_moves_from_info(engine, b_best, depth, n_plies)
    if not played_moves and not best_moves:
        return None

    played_leaf_cp = _leaf_eval_after_line(engine, b_played, played_moves, depth)
    best_leaf_cp = _leaf_eval_after_line(engine, b_best, best_moves, depth)

    # Build boards at leaf
    bp = b_played.copy()
    for m in played_moves:
        if m not in bp.legal_moves:
            break
        bp.push(m)
    bq = b_best.copy()
    for m in best_moves:
        if m not in bq.legal_moves:
            break
        bq.push(m)

    try:
        f_played = compute_hidden_features(bp)
        f_best = compute_hidden_features(bq)
    except Exception as e:
        logger.warning("future_line_compare: hidden features failed: %s", e)
        f_played = {}
        f_best = {}

    feat_delta = _top_feature_deltas(f_played, f_best)
    gap = None
    if played_leaf_cp is not None and best_leaf_cp is not None:
        gap = best_leaf_cp - played_leaf_cp

    return FutureLineDelta(
        played_leaf_eval_cp=played_leaf_cp,
        best_leaf_eval_cp=best_leaf_cp,
        eval_gap_cp=gap,
        feature_deltas=feat_delta,
        played_targets=_destination_targets(played_moves, min_count=2),
        best_targets=_destination_targets(best_moves, min_count=2),
        played_line_san=_moves_to_san(b_played, played_moves),
        best_line_san=_moves_to_san(b_best, best_moves),
    )


def _moves_to_san(start: chess.Board, moves: List[chess.Move]) -> List[str]:
    b = start.copy()
    out: List[str] = []
    for m in moves:
        if m not in b.legal_moves:
            break
        out.append(b.san(m))
        b.push(m)
    return out
