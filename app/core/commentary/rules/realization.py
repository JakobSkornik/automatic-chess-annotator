"""Per-ply feature progression along a line (for the charts). Claim realization
(immediate vs envisioned) is set from the firing horizon in facts_builder."""

from __future__ import annotations

from app.core.commentary.features.guid_features import compute_feature_vector_fen


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
