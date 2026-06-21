"""Low-level helpers for the engine pass: PV/eval extraction from an engine
info blob, and the strategic before/after feature-delta map."""

from __future__ import annotations

from typing import Any

from .constants import MATE_SCORE


def _cp_and_pv1(info_any: Any) -> tuple[int | None, str | None]:
    """White-POV centipawns and the first PV move (UCI) from an engine info blob."""
    inf = info_any[0] if isinstance(info_any, list) and info_any else info_any
    if not isinstance(inf, dict):
        return None, None
    sc = inf.get("score")
    if sc is None:
        return None, None
    try:
        cp = int(sc.white().score(mate_score=MATE_SCORE))
    except Exception:
        return None, None
    pv = inf.get("pv") or []
    uci = pv[0].uci() if pv else None
    return cp, uci


def _compute_features_delta(before: dict, after: dict) -> dict:
    """Strategic before/after deltas (open files, pawn counts, boolean flags)."""
    delta: dict = {"white": {}, "black": {}, "openFiles": {}}
    try:
        # Open files count deltas
        if isinstance(before.get("openFiles"), dict) and isinstance(
            after.get("openFiles"), dict
        ):
            try:
                delta["openFiles"]["openCount"] = len(
                    after["openFiles"].get("open", [])
                ) - len(before["openFiles"].get("open", []))
                delta["openFiles"]["semiOpenWhiteCount"] = len(
                    after["openFiles"].get("semiOpenWhite", [])
                ) - len(before["openFiles"].get("semiOpenWhite", []))
                delta["openFiles"]["semiOpenBlackCount"] = len(
                    after["openFiles"].get("semiOpenBlack", [])
                ) - len(before["openFiles"].get("semiOpenBlack", []))
            except Exception:
                pass

        def _list_or_int_len(d: dict, key: str) -> int:
            v = d.get(key)
            if isinstance(v, list):
                return len(v)
            if isinstance(v, int):
                return v
            return 0

        for side in ("white", "black"):
            b = before.get(side, {}) if isinstance(before.get(side, {}), dict) else {}
            a = after.get(side, {}) if isinstance(after.get(side, {}), dict) else {}

            def diff_num(key: str, b=b, a=a, side=side):
                if isinstance(b.get(key), int) and isinstance(a.get(key), int):
                    delta[side][key] = a[key] - b[key]

            for k in ("doubledPawns", "isolatedPawns", "passedPawns"):
                delta[side][k] = _list_or_int_len(a, k) - _list_or_int_len(b, k)
            for k in (
                "attackedPieces",
                "attackingPieces",
                "rooksOnOpenFiles",
                "rooksOnSemiOpenFiles",
            ):
                diff_num(k)
            # Booleans as changed flags
            for k in (
                "hasBishopPair",
                "canCastleKingSide",
                "canCastleQueenSide",
                "connectedRooks",
            ):
                if (
                    isinstance(b.get(k), bool)
                    and isinstance(a.get(k), bool)
                    and a[k] != b[k]
                ):
                    delta[side][f"{k}Changed"] = True
        return delta
    except Exception:
        return {}
