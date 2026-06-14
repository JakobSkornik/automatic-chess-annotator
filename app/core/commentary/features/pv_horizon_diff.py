"""Compare root hidden features vs position after walking engine PV (White POV static metrics)."""

from __future__ import annotations

import logging
from typing import Any

import chess

from app.core.commentary.features.positional_features import compute_hidden_features
from app.core.engine.engine_connector import EngineConnector
from app.models.chess_events import PvHorizonDiff

logger = logging.getLogger(__name__)

MATE_SCORE = 1_000_000

_SCALAR_KEYS = (
    "mobility",
    "centerControl",
    "space",
    "centralization",
    "kingExposure",
    "kingShieldPawns",
    "kingZoneAttacks",
    "attackedPieces",
    "attackingPieces",
)

_LIST_KEYS_PER_SIDE = (
    "doubledPawns",
    "isolatedPawns",
    "backwardPawns",
    "weakSquares",
    "holes",
)


def _normalize_info(info: Any) -> dict | None:
    if isinstance(info, list) and info:
        info = info[0]
    return info if isinstance(info, dict) else None


def _info_to_white_cp(info: Any) -> int | None:
    inf = _normalize_info(info)
    if not inf:
        return None
    sc = inf.get("score")
    if sc is None:
        return None
    try:
        return int(sc.white().score(mate_score=MATE_SCORE))
    except Exception:
        return None


def _sq_strings_from_side(side: dict[str, Any], key: str) -> set[str]:
    v = side.get(key)
    if not isinstance(v, list):
        return set()
    if key == "passedPawns":
        out: set[str] = set()
        for item in v:
            if isinstance(item, dict) and item.get("sq"):
                out.add(str(item["sq"]))
        return out
    return {str(x) for x in v if isinstance(x, str)}


def _contested_sq_set(feat: dict[str, Any]) -> set[str]:
    raw = feat.get("contestedSquares")
    if not isinstance(raw, list):
        return set()
    out: set[str] = set()
    for item in raw:
        if isinstance(item, dict) and item.get("sq"):
            out.add(str(item["sq"]))
    return out


def _scalar_flatten(feat: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for side in ("white", "black"):
        d = feat.get(side)
        if not isinstance(d, dict):
            continue
        for k in _SCALAR_KEYS:
            v = d.get(k)
            if isinstance(v, (int, float)):
                out[f"{side}.{k}"] = float(v)
            elif v is None:
                continue
    return out


def _list_deltas_between(
    root: dict[str, Any], leaf: dict[str, Any]
) -> dict[str, dict[str, list[str]]]:
    deltas: dict[str, dict[str, list[str]]] = {}
    for side in ("white", "black"):
        r0 = root.get(side)
        r1 = leaf.get(side)
        if not isinstance(r0, dict) or not isinstance(r1, dict):
            continue
        for key in _LIST_KEYS_PER_SIDE:
            s0 = _sq_strings_from_side(r0, key)
            s1 = _sq_strings_from_side(r1, key)
            added = sorted(s1 - s0)
            removed = sorted(s0 - s1)
            label = f"{side}.{key}"
            if added or removed:
                deltas[label] = {"added": added, "removed": removed}
    c0 = _contested_sq_set(root)
    c1 = _contested_sq_set(leaf)
    added_c = sorted(c1 - c0)
    removed_c = sorted(c0 - c1)
    if added_c or removed_c:
        deltas["contestedSquares"] = {"added": added_c, "removed": removed_c}
    return deltas


def compute_pv_horizon_diff(
    engine: EngineConnector,
    fen: str,
    *,
    plies: int = 10,
    depth: int = 18,
) -> PvHorizonDiff | None:
    """
    Walk multipv=1 PV up to ``plies`` half-moves from ``fen``, compare ``compute_hidden_features``
    at root vs leaf (same densification rules). Returns None if PV too short or analysis fails.
    """
    try:
        board = chess.Board(fen)
    except Exception as e:
        logger.debug("pv_horizon_diff: bad fen %s", e)
        return None

    info = _normalize_info(engine.analyse(board, depth=depth, multiPv=1))
    if not info:
        return None
    pv = info.get("pv") or []
    if len(pv) < 4:
        return None
    root_feat = compute_hidden_features(board)
    if not isinstance(root_feat, dict):
        return None

    n_walk = min(plies, len(pv))
    b = board.copy()
    san_line: list[str] = []
    for i in range(n_walk):
        m = pv[i]
        if m not in b.legal_moves:
            logger.debug("pv_horizon_diff: illegal PV step %s at ply %s", m, i)
            return None
        san_line.append(b.san(m))
        b.push(m)

    leaf_feat = compute_hidden_features(b)
    if not isinstance(leaf_feat, dict):
        return None

    leaf_cp = _info_to_white_cp(engine.analyse(b, depth=depth, multiPv=1))

    a = _scalar_flatten(root_feat)
    bmap = _scalar_flatten(leaf_feat)
    keys = set(a) | set(bmap)
    scalar_deltas: dict[str, float] = {}
    for k in sorted(keys):
        va = a.get(k, 0.0)
        vb = bmap.get(k, 0.0)
        diff = round(vb - va, 3)
        if diff != 0.0:
            scalar_deltas[k] = diff

    list_deltas = _list_deltas_between(root_feat, leaf_feat)

    return PvHorizonDiff(
        plies=n_walk,
        pv_san=san_line,
        leaf_eval_cp=leaf_cp,
        scalar_deltas=scalar_deltas,
        list_deltas=list_deltas,
    )
