"""Tunable thresholds and limits for the rule-based Expert Module (centipawns).

Hand-tuned per Guid §5.4.2."""

THRESHOLDS: dict[str, int] = {
    "pawn_structure_total": 14,  # Z in the dissertation's pawn rule
    "pawn_structure_evaluate": 8,  # Y — net EVALUATE_PAWNS shift
    "doubled_pawns": 12,
    "strong_knight": 15,
    "rook_activity": 15,
    "king_safety": 18,
    "king_tropism_corroborate": 8,
    "piece_activity": 14,
    "center_control": 16,
    "space": 10,
    "bishop_color_complex": 12,
    "king_activity_endgame": 10,
    "passer_escort": 8,
    "bad_bishop": 25,  # min |delta| before a bad-bishop change is worth stating
    "connected_rooks": 10,
    "passer_advance": 15,
    "min_claim_cp": 8,  # ignore fired rules weaker than this
}

SIDES = ("WHITE", "BLACK")

MAX_CONCESSIONS = 2
MAX_CLAIMS_PER_MOVE = 4
MAX_ALTERNATIVE_MERITS = 2

_MATE_SCORE = 1000000
# Beyond this eval the game is decided, so positional claims are dropped as noise.
DECISIVE_CLAIM_CP = 500
# When the move leaves the mover at least this much worse (mover-POV cp), its
# incidental positional "merits" are misleading consolation and are dropped —
# only the consequences that explain the result are kept.
MERIT_SUPPRESS_CP = 100
MATERIAL_LOSS_CP = 100  # mover-POV material drop that earns a "loses material" claim
EVAL_CONCESSION_CP = 60  # weight for the "engine preferred X" fallback claim

# A claim is "immediate" when at least this fraction of its total (start->leaf)
# feature swing has already happened by the position right after the move.
REALIZE_FRACTION = 0.5
REALIZE_EPS_CP = 10  # below this magnitude there is no swing worth deferring
