"""Serialization of CommentFacts and per-ply feature series into GameJson dicts."""

from __future__ import annotations

from typing import Any

from app.core.commentary.features.guid_features import CHART_FEATURES
from app.models.chess_events import AnalyzedMoveData
from app.models.GameJson import FeatureSeries


def _facts_to_json(facts: Any) -> dict[str, Any]:
    """Trimmed CommentFacts for the structured comment renderer in the UI."""

    def _line(dl: Any) -> dict[str, Any] | None:
        if dl is None or not getattr(dl, "line_san", None):
            return None
        return {
            "start_fen": dl.start_fen,
            "san": list(dl.line_san),
            "fens": list(dl.fens),
            "feature_series": dict(getattr(dl, "feature_series", {}) or {}),
        }

    def _claims(claims: Any) -> list[dict[str, Any]]:
        return [
            {
                "text": c.text,
                "text_state": c.text_state,
                "features": list(c.features_involved),
                "delta_cp": c.delta_cp,
                "flag_note": c.flag_note,
                "beneficiary": c.beneficiary,
                "is_concession": c.is_concession,
                "realization": c.realization,
            }
            for c in (claims or [])
        ]

    out: dict[str, Any] = {
        "verdict": facts.verdict,
        "eval_cp": facts.eval_cp,
        "eval_mate": facts.eval_mate,
        "depth": facts.depth,
        "engine": facts.engine,
        "display_line": _line(facts.display_line),
        "claims": _claims(facts.claims),
        "concession_mode": facts.concession_mode,
    }
    alt = facts.better_alternative
    if alt is not None:
        out["better_alternative"] = {
            "san": alt.san,
            "uci": alt.uci,
            "verdict": alt.verdict,
            "eval_cp": alt.eval_cp,
            "display_line": _line(alt.display_line),
            "claims": _claims(alt.claims),
        }
    return out


def _build_feature_series(analyzed_rows: list[AnalyzedMoveData]) -> FeatureSeries:
    """Aligned per-ply arrays of the charted Guid features (White-POV cp)."""
    plies: list[int] = []
    by_name: dict[str, list[int | None]] = {name: [] for name in CHART_FEATURES}
    for row in analyzed_rows:
        plies.append(row.ply)
        guid = (row.hidden_features or {}).get("_guid") or {}
        for name in CHART_FEATURES:
            d = guid.get(name)
            if isinstance(d, dict) and d.get("v") is not None:
                by_name[name].append(int(d["v"]))
            else:
                by_name[name].append(None)
    return FeatureSeries(plies=plies, features=by_name)
