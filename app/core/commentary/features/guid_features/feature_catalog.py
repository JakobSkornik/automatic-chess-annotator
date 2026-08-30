"""Single source of truth for feature *presentation* metadata.

Each positional feature is described once here — its display label, natural unit,
group, provenance (Stockfish definition vs our own), and a short human
explanation (adapted from the Stockfish Evaluation Guide where the feature has a
Stockfish equivalent). The frontend's feature labels/units/descriptions are
generated from this catalog (see ``scripts/export_feature_meta.py``), so the two
never drift.

Computation lives separately (the feature modules / ``vector.py``); this module
holds only metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Unit = Literal["cp", "squares", "pawns", "count", "flag"]
Source = Literal["stockfish", "custom"]


@dataclass(frozen=True)
class FeatureSpec:
    """Presentation metadata for one feature (keyed by its base name)."""

    name: str  # base name without WHITE_/BLACK_ prefix, or a net composite name
    label: str  # short display label
    group: str
    unit: Unit
    source: Source
    description: str
    chart: bool = True


_SPECS: tuple[FeatureSpec, ...] = (
    # --- material ---
    FeatureSpec(
        "MATERIAL_BALANCE",
        "Material",
        "Material",
        "cp",
        "stockfish",
        "Net material on the board (White minus Black), in centipawns.",
    ),
    # --- pawns (Stockfish pawn structure) ---
    FeatureSpec(
        "EVALUATE_PAWNS",
        "Pawn Structure",
        "Pawns",
        "cp",
        "custom",
        "Overall pawn-structure balance: doubled, isolated, backward and passed pawns.",
    ),
    FeatureSpec(
        "PAWN_DOUBLED",
        "Doubled Pawns",
        "Pawns",
        "pawns",
        "stockfish",
        "Doubled pawns: a pawn directly behind a friendly pawn on the same file, "
        "unsupported.",
    ),
    FeatureSpec(
        "PAWN_ISOLATED",
        "Isolated Pawns",
        "Pawns",
        "pawns",
        "stockfish",
        "Isolated pawns: no friendly pawn on either adjacent file.",
    ),
    FeatureSpec(
        "PAWN_BACKWARD",
        "Backward Pawns",
        "Pawns",
        "pawns",
        "stockfish",
        "Backward pawns: behind the pawns on adjacent files and unable to advance "
        "safely.",
    ),
    FeatureSpec(
        "WEAK_PAWNS",
        "Weak Pawns",
        "Pawns",
        "pawns",
        "stockfish",
        "Weak pawns (isolated or backward) that are hard to defend.",
    ),
    FeatureSpec(
        "PAWN_PASSED",
        "Passed Pawns",
        "Pawns",
        "cp",
        "stockfish",
        "Passed pawns: no enemy pawn can stop them on their file or the adjacent "
        "files.",
    ),
    FeatureSpec(
        "PAWN_DUO",
        "Pawn Phalanx",
        "Pawns",
        "pawns",
        "stockfish",
        "Phalanx: two friendly pawns side by side on adjacent files.",
    ),
    FeatureSpec(
        "PAWN_ADVANCES",
        "Pawn Advancement",
        "Pawns",
        "cp",
        "custom",
        "How far the pawns have advanced from their starting rank.",
    ),
    # --- knights ---
    FeatureSpec(
        "KNIGHTS_OUTPOSTS",
        "Knight Outposts",
        "Pieces",
        "count",
        "stockfish",
        "Knights on an outpost square — supported by a pawn and immune to enemy "
        "pawn attack.",
    ),
    FeatureSpec(
        "KNIGHTS_CENTRALIZATION",
        "Knight Centralization",
        "Pieces",
        "cp",
        "custom",
        "How centrally the knights are posted, where they influence most squares.",
    ),
    # --- bishops ---
    FeatureSpec(
        "BISHOP_PAIR",
        "Bishop Pair",
        "Pieces",
        "cp",
        "stockfish",
        "Bonus for holding both bishops, which cover complementary colour complexes.",
    ),
    FeatureSpec(
        "BISHOP_PLUS_PAWNS_ON_COLOR",
        "Bishop vs Own Pawns",
        "Pieces",
        "count",
        "stockfish",
        "Own pawns on the bishop's colour, scaled by blocked central pawns "
        "(Stockfish bishop-pawns) — higher means the bishop is more hemmed in.",
    ),
    FeatureSpec(
        "BISHOPS_MOBILITY",
        "Bishop Mobility",
        "Pieces",
        "cp",
        "stockfish",
        "How many squares the bishops can reach — their scope and activity.",
    ),
    FeatureSpec(
        "BAD_BISHOP",
        "Bad Bishop",
        "Pieces",
        "cp",
        "custom",
        "A bishop hemmed in by its own pawns fixed on its colour, with low mobility.",
    ),
    # --- rooks ---
    FeatureSpec(
        "ROOK_OPEN_FILE",
        "Rook on Open File",
        "Pieces",
        "count",
        "stockfish",
        "Rooks on a file with no pawns of either colour.",
    ),
    FeatureSpec(
        "ROOK_HALF_OPEN_FILE",
        "Rook on Half-Open File",
        "Pieces",
        "count",
        "stockfish",
        "Rooks on a file with no friendly pawns but an enemy pawn.",
    ),
    FeatureSpec(
        "ROOK_ON_SEVENTH",
        "Rook on 7th",
        "Pieces",
        "cp",
        "custom",
        "Rook(s) on the 7th rank, attacking pawns and confining the enemy king.",
    ),
    FeatureSpec(
        "ROOK_BEHIND_PASSED_PAWN",
        "Rook Behind Passer",
        "Pieces",
        "cp",
        "custom",
        "Rook supporting or blockading a passed pawn from behind.",
    ),
    FeatureSpec(
        "ROOKS_CONNECTED",
        "Connected Rooks",
        "Pieces",
        "cp",
        "custom",
        "Rooks defending each other along a rank or file.",
    ),
    # --- activity / space ---
    FeatureSpec(
        "PIECE_ACTIVITY",
        "Piece Activity",
        "Activity",
        "squares",
        "stockfish",
        "Piece mobility: squares the minor/major pieces can reach in the mobility "
        "area (Stockfish definition). Measured in squares.",
    ),
    FeatureSpec(
        "CENTER_CONTROL",
        "Center Control",
        "Activity",
        "cp",
        "custom",
        "Control of the central squares (d4, e4, d5, e5) by pawns and pieces.",
    ),
    FeatureSpec(
        "SPACE",
        "Space",
        "Activity",
        "cp",
        "stockfish",
        "Space advantage: safe squares controlled behind the pawn chain.",
    ),
    # --- king safety ---
    FeatureSpec(
        "EVALUATE_KING_SAFETY",
        "King Safety",
        "King",
        "cp",
        "custom",
        "Overall king safety: pawn shelter, attacking pieces and exposed squares "
        "around the king.",
    ),
    FeatureSpec(
        "KING_DANGER",
        "King Danger",
        "King",
        "cp",
        "stockfish",
        "Pressure on the king (Stockfish king-danger core): enemy attackers and "
        "their weight, attacks next to the king, and weakly-defended squares "
        "around it. Negative for the side whose king is under fire.",
    ),
    FeatureSpec(
        "KING_ZONE_ATTACKS",
        "King Zone Attacks",
        "King",
        "count",
        "stockfish",
        "Minor/major pieces aiming at the enemy king's zone — an attack-arc "
        "counter that registers pressure before it becomes outright danger.",
    ),
    FeatureSpec(
        "BACK_RANK",
        "Back-Rank Weakness",
        "King",
        "cp",
        "custom",
        "Risk of back-rank mate: a king hemmed in by its own pawns with no luft.",
    ),
    FeatureSpec(
        "KING_TROPISM",
        "King Tropism",
        "King",
        "cp",
        "custom",
        "How close the pieces are to the enemy king — a proxy for attacking pressure.",
    ),
    FeatureSpec(
        "CASTLING_RIGHTS",
        "Castling Rights",
        "King",
        "cp",
        "custom",
        "Retained castling rights — the flexibility to still castle either side.",
    ),
    # --- threats ---
    FeatureSpec(
        "WEAK_ENEMIES",
        "Weak Enemies",
        "Threats",
        "count",
        "stockfish",
        "Enemy pieces under our attack and not defended by a pawn.",
    ),
    FeatureSpec(
        "HANGING",
        "Hanging Enemies",
        "Threats",
        "count",
        "stockfish",
        "Weak enemy pieces that are undefended, or non-pawns we attack more than "
        "once — pieces in real danger of being won.",
    ),
    FeatureSpec(
        "PINS",
        "Pins",
        "Threats",
        "count",
        "stockfish",
        "Enemy minor/major pieces pinned to their king by our sliders.",
    ),
    # --- endgame ---
    FeatureSpec(
        "KING_ACTIVITY",
        "King Activity",
        "Endgame",
        "cp",
        "custom",
        "How actively the king takes part — mostly relevant in the endgame.",
    ),
    FeatureSpec(
        "OUTSIDE_PASSER",
        "Outside Passer",
        "Endgame",
        "cp",
        "custom",
        "A passed pawn far from the kings — often decisive in the endgame.",
    ),
    FeatureSpec(
        "PASSER_KING_ESCORT",
        "Passer King Escort",
        "Endgame",
        "cp",
        "stockfish",
        "Own king close to a friendly passed pawn, helping to escort it.",
    ),
)

CATALOG: dict[str, FeatureSpec] = {spec.name: spec for spec in _SPECS}
