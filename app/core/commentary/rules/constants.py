"""Tunable thresholds and limits for the rule-based Expert Module (centipawns).

Hand-tuned per Guid §5.4.2."""

THRESHOLDS: dict[str, int] = {
    "pawn_structure_evaluate": 8,  # net EVALUATE_PAWNS (cp) shift for the structure rule
    "doubled_pawns": 12,
    "strong_knight": 1,  # count: a knight reaching/leaving an outpost
    "rook_activity": 1,  # count: a rook reaching an open/semi-open file
    "king_safety": 60,  # king-danger swing (Stockfish scale) marking a real shift
    "king_tropism_corroborate": 8,
    "piece_activity": 7,  # mobility-square count (not cp): a 7-square swing
    "center_control": 16,
    "space": 10,
    "bishop_color_complex": 12,
    "king_activity_endgame": 10,
    "passer_escort": 8,
    "bad_bishop": 25,  # min |delta| before a bad-bishop change is worth stating
    "connected_rooks": 10,
    "passer_advance": 15,
    "min_claim_cp": 8,  # ignore fired rules weaker than this
    "material_standing": 200,  # min standing edge (cp) to restate when unchanged
}

# Natural-count features (mobility squares, pawn counts) store their value in
# their own unit. This maps one unit of change to a cp-comparable claim
# importance, so a count-feature claim ranks and clears `min_claim_cp` alongside
# the centipawn-scored features. Used for claim weighting only — never the eval.
COUNT_CLAIM_CP: dict[str, int] = {
    "PIECE_ACTIVITY": 3,  # per mobility square
    "PAWN_DOUBLED": 12,  # per doubled pawn
    "PAWN_ISOLATED": 10,
    "PAWN_BACKWARD": 8,
    "WEAK_PAWNS": 9,
    "PAWN_DUO": 4,
    "KNIGHTS_OUTPOSTS": 18,  # per knight outpost
    "ROOK_OPEN_FILE": 20,  # per rook on an open file
    "ROOK_HALF_OPEN_FILE": 10,  # per rook on a semi-open file
    "BISHOP_PLUS_PAWNS_ON_COLOR": 4,  # per unit of the bishop-pawns score
    "HANGING": 60,  # per hanging enemy piece (a concrete material threat)
    "WEAK_ENEMIES": 15,  # per weak enemy piece
    "PINS": 25,  # per pinned enemy piece
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
# For a played best move, the runner-up reads as clearly "weaker" at or above
# this mover-POV gap; below it, as "a comparable alternative" (Guid).
INFERIOR_ALT_WEAKER_CP = 60
