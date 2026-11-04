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
            "leaf_fen": dl.leaf_fen,
            "start_quiescent": dl.start_quiescent,
            "leaf_quiescent": dl.leaf_quiescent,
            "evidence_valid": dl.evidence_valid,
            "trimmed_plies": getattr(dl, "trimmed_plies", 0),
            "immediate_horizon_ply": getattr(dl, "immediate_horizon_ply", 1),
            "settled_ply": getattr(dl, "settled_ply", None),
            "stable_ply": getattr(dl, "stable_ply", None),
            "horizon_reason": getattr(dl, "horizon_reason", None),
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
                "rule_id": c.rule_id,
                "support_ply": c.support_ply,
                "support_fen": c.support_fen,
                "baseline_fen": c.baseline_fen,
                "caused_by_ply": getattr(c, "caused_by_ply", None),
                "caused_by_san": getattr(c, "caused_by_san", None),
                "first_ply": getattr(c, "first_ply", None),
                "last_ply": getattr(c, "last_ply", None),
                "persistence": getattr(c, "persistence", "held"),
                "entity": getattr(c, "entity", None),
                "evidence": (c.evidence.model_dump() if getattr(c, "evidence", None) is not None else None),
            }
            for c in (claims or [])
        ]

    out: dict[str, Any] = {
        "verdict": facts.verdict,
        "eval_cp": facts.eval_cp,
        "eval_mate": facts.eval_mate,
        "terminal_result": facts.terminal_result,
        "depth": facts.depth,
        "engine": facts.engine,
        "display_line": _line(facts.display_line),
        "claims": _claims(facts.claims),
        "concession_mode": facts.concession_mode,
        "selection_metadata": facts.selection_metadata,
        "best_margin_cp": getattr(facts, "best_margin_cp", None),
        "move_quality": getattr(facts, "move_quality", None),
    }
    alt = facts.better_alternative
    if alt is not None:
        out["better_alternative"] = {
            "san": alt.san,
            "uci": alt.uci,
            "verdict": alt.verdict,
            "eval_cp": alt.eval_cp,
            "eval_mate": alt.eval_mate,
            "display_line": _line(alt.display_line),
            "claims": _claims(alt.claims),
            "is_inferior": alt.is_inferior,
            "feature_advantages": [a.model_dump() for a in alt.feature_advantages],
            "matched_ply": getattr(alt, "matched_ply", None),
            "feature_advantages_matched": [a.model_dump() for a in getattr(alt, "feature_advantages_matched", [])],
            "horizon_note": getattr(alt, "horizon_note", None),
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
