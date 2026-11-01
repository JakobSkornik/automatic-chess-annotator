"""Tunable thresholds and limits for the rule-based Expert Module (centipawns).

Originally hand-tuned per Guid §5.4.2. The feature thresholds marked "1 Nov 2026" were
set from the distribution of feature changes on the ten survey games, so that each
rule fires in roughly 5-10% of positions (see THRESHOLD-ANALYSIS-20261101.md)."""

THRESHOLDS: dict[str, int] = {
    "pawn_structure_evaluate": 25,  # 1 Nov 2026 (was 8: fired on 49% of positions)  # net EVALUATE_PAWNS (cp) shift for the structure rule
    "doubled_pawns": 12,
    "strong_knight": 1,  # count: a knight reaching/leaving an outpost
    "rook_activity": 1,  # count: a rook reaching an open/semi-open file
    "king_safety": 400,  # 1 Nov 2026 (was 60 with a tropism condition): king-danger swing
    "piece_activity": 10,  # 1 Nov 2026 (was 7): mobility-square count (not cp)
    "center_control": 12,  # 1 Nov 2026 (was 16)
    "space": 6,  # 1 Nov 2026 (was 10: never reached)
    "bishop_color_complex": 8,  # 1 Nov 2026 (was 12, read by no rule until then)
    "king_activity_endgame": 5,  # 1 Nov 2026 (was 10)
    "passer_escort": 4,  # 1 Nov 2026 (was 8)
    "bad_bishop": 20,  # 1 Nov 2026 (was 25; the change is always 20)
    "connected_rooks": 8,  # 1 Nov 2026 (was 10; the change is always 8)
    "passer_advance": 5,  # 1 Nov 2026 (was 15)
    "king_zone_attacks": 2,  # 1 Nov 2026: extra attackers on the enemy king zone
    "pawn_advances": 4,  # 1 Nov 2026: pawn-advance score (75th percentile)
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
# Feature-diff-of-diffs (Improvement 3): how many top per-feature advantages
# to keep on a BestAlternative, and how big the cp gap has to be to bother
# surfacing at all -- reuses THRESHOLDS["min_claim_cp"] as the materiality
# bar so a feature-advantage is never "smaller" than what would already earn
# a claim on its own.
MAX_FEATURE_ADVANTAGES = 3

_MATE_SCORE = 1000000
# Beyond this eval the game is decided, so positional claims are dropped as noise.
# Tightened from 500 to 400 per project advisor feedback: avoid commenting on
# missed alternatives/positional nuance once a position is decisively won or
# lost (the advisor's own published methodology used a +-200cp interval and
# suggested this codebase could reasonably use +-300..+-500cp; 400 is the
# midpoint of that suggested range).
DECISIVE_CLAIM_CP = 400
# When the move leaves the mover at least this much worse (mover-POV cp), its
# incidental positional "merits" are misleading consolation and are dropped —
# only the consequences that explain the result are kept. Raised from 100 to
# 150 (audit: facts_builder.py step-1 finding): 100cp is only one pawn worse,
# a routine and often still-playable concession after which the mover's own
# genuine merits (e.g. gaining the bishop pair) are not actually misleading —
# this only starts dropping merits once the mover is clearly, not just
# marginally, worse off.
MERIT_SUPPRESS_CP = 150
MATERIAL_LOSS_CP = 100  # mover-POV material drop that earns a "loses material" claim
EVAL_CONCESSION_CP = 60  # weight for the "engine preferred X" fallback claim
# For a played best move, the runner-up reads as clearly "weaker" at or above
# this mover-POV gap; below it, as "a comparable alternative" (Guid).
INFERIOR_ALT_WEAKER_CP = 60
