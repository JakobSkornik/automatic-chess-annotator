"""Map PvHorizonDiff feature deltas to strategic motif labels."""

from __future__ import annotations

import chess

from app.models.chess_events import PvHorizonDiff, StrategicMotif

_KING_EXPOSURE_THRESHOLD = 2.0
_MOBILITY_COLLAPSE_THRESHOLD = 4.0
_CENTER_CONTROL_SHIFT = 2.0
_SPACE_GAIN = 3.0
_ATTACKING_PIECES_GAIN = 2.0


def _side_from_key(key: str) -> str:
    if key.startswith("white."):
        return "white"
    if key.startswith("black."):
        return "black"
    return ""


def infer_motifs_from_deltas(
    diff: PvHorizonDiff | None,
    *,
    mover: chess.Color | None = None,
    phase: str = "middlegame",
) -> list[StrategicMotif]:
    """
    Translate scalar/list deltas from a PV horizon walk into strategic motif tags.
    ``mover`` is the side that played the move at the root (for directional interpretation).
    """
    if diff is None:
        return []

    motifs: list[StrategicMotif] = []
    seen: set[StrategicMotif] = set()
    scalars = diff.scalar_deltas or {}
    lists = diff.list_deltas or {}

    def _add(m: StrategicMotif) -> None:
        if m not in seen:
            seen.add(m)
            motifs.append(m)

    for key, delta in scalars.items():
        side = _side_from_key(key)
        metric = key.split(".", 1)[-1] if "." in key else key

        if metric == "kingExposure" and delta >= _KING_EXPOSURE_THRESHOLD:
            if mover is not None:
                enemy = chess.BLACK if mover == chess.WHITE else chess.WHITE
                enemy_label = "black" if enemy == chess.BLACK else "white"
                if side == enemy_label:
                    _add(StrategicMotif.WEAK_SQUARE_EXPLOITED)
            else:
                _add(StrategicMotif.WEAK_SQUARE_EXPLOITED)

        if metric == "mobility" and delta <= -_MOBILITY_COLLAPSE_THRESHOLD:
            _add(StrategicMotif.RESTRICTION)

        if metric == "space" and delta >= _SPACE_GAIN:
            _add(StrategicMotif.SPACE_ADVANTAGE)

        if metric == "centerControl" and abs(delta) >= _CENTER_CONTROL_SHIFT:
            _add(StrategicMotif.PAWN_LEVER)

        if metric == "attackingPieces" and delta >= _ATTACKING_PIECES_GAIN:
            _add(StrategicMotif.DOMINATION)

        if metric == "kingZoneAttacks" and delta >= 2.0:
            _add(StrategicMotif.WEAK_SQUARE_CREATION)

    for label, changes in lists.items():
        added = changes.get("added") or []
        removed = changes.get("removed") or []

        if "passedPawns" in label and added:
            if phase == "endgame":
                _add(StrategicMotif.OUTSIDE_PASSER)
            else:
                _add(StrategicMotif.PASSED_PAWN_MIDDLEGAME)

        if "isolatedPawns" in label and added:
            _add(StrategicMotif.ISOLATED_QUEEN_PAWN)

        if "doubledPawns" in label and added:
            _add(StrategicMotif.HANGING_PAWNS)

        if "weakSquares" in label and added:
            _add(StrategicMotif.WEAK_SQUARE_CREATION)

        if label == "contestedSquares" and added:
            _add(StrategicMotif.PAWN_LEVER)

        if "backwardPawns" in label and added:
            _add(StrategicMotif.BACKWARD_PAWN_TARGET)

        if "holes" in label and added:
            _add(StrategicMotif.WEAK_SQUARE_CREATION)

        if "passedPawns" in label and removed and phase != "endgame":
            _add(StrategicMotif.SIMPLIFICATION_WHEN_AHEAD)

    return motifs
