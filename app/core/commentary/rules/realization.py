"""Per-ply feature progression along a line, and tagging each claim as realized
immediately on the move vs only developing later in the variation."""

from __future__ import annotations

from app.core.commentary.features.guid_features import compute_feature_vector_fen
from app.models.comment_facts import Claim

from .constants import REALIZE_EPS_CP, REALIZE_FRACTION


def _line_feature_series(
    start_fen: str, fens: list[str], names: list[str] | None = None
) -> dict:
    """Per-point White-POV cp series along a line (point 0 = start position, then
    one per kept ply). Engine-free. ``names=None`` emits every feature so each
    chart in the navigator has both the game and the along-the-line curve."""
    points = [start_fen, *list(fens)]
    vecs = []
    for fen in points:
        try:
            vecs.append(compute_feature_vector_fen(fen) if fen else None)
        except Exception:
            vecs.append(None)
    if names is None:
        keys: set[str] = set()
        for vec in vecs:
            if vec:
                keys.update(vec.keys())
        names = sorted(keys)
    if not names:
        return {}
    series: dict = {name: [] for name in names}
    for vec in vecs:
        for name in names:
            fv = vec.get(name) if vec else None
            if fv is not None:
                series[name].append(fv.value_cp)
            else:
                series[name].append(series[name][-1] if series[name] else 0)
    return series


# A claim is "immediate" when at least this fraction of its total (start->leaf)
# feature swing has already happened by the position right after the move; below
# it, the change only develops deeper in the line ("envisioned").


def _claim_realization(claim: Claim, feature_series: dict[str, list[int]]) -> str:
    """immediate vs envisioned, judged on the claim's dominant feature trajectory.

    ``feature_series[name]`` is the per-point White-POV cp series along the line
    (point 0 = start, point 1 = right after the move, last = envisioned leaf)."""
    dom: list[int] | None = None
    dom_total = 0
    for name in claim.features_involved:
        series = feature_series.get(name)
        if not series or len(series) < 2:
            continue
        total = series[-1] - series[0]
        if dom is None or abs(total) > abs(dom_total):
            dom, dom_total = series, total
    if dom is None or len(dom) < 2:
        return "immediate"
    total = dom[-1] - dom[0]
    moved = dom[1] - dom[0]
    if abs(total) < REALIZE_EPS_CP:
        return "immediate"
    same_sign = (moved >= 0) == (total >= 0)
    if same_sign and abs(moved) >= REALIZE_FRACTION * abs(total):
        return "immediate"
    return "envisioned"


def _annotate_realization(
    claims: list[Claim], feature_series: dict[str, list[int]]
) -> None:
    for c in claims:
        c.realization = _claim_realization(c, feature_series)
