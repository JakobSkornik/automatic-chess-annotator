"""Rule-based Expert Module (Guid §5.4).

Rules are threshold conditions over the feature-difference vector between
the starting position and the envisioned position. Each fired rule yields a
``Claim``: one declarative sentence plus the feature names backing it (which
the UI uses to highlight the matching charts). If no rule fires and the move
is not a quality-triggered key moment, the move stays uncommented — Guid's
"better silent than wrong".

All thresholds live in ``THRESHOLDS`` (centipawns) and are meant to be tuned
by hand (Guid §5.4.2).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import chess

from app.core.commentary.features.envisioned import (
    diff_vectors,
    envisioned_for_best_move,
    envisioned_for_played_move,
)
from app.core.commentary.features.guid_features import (
    bad_bishop_squares,
    compute_feature_vector_fen,
    doubled_pawn_files,
    outpost_squares,
    passed_pawn_squares,
    rooks_on_seventh_squares,
)
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.comment_facts import (
    BestAlternative,
    Claim,
    CommentFacts,
    FeatureDelta,
    FeatureDiff,
)

logger = logging.getLogger(__name__)

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
    "min_claim_cp": 8,  # ignore fired rules weaker than this
    "better_alternative_gap": 50,  # cp loss before the best move is shown
}

SIDES = ("WHITE", "BLACK")


def _toward(side: str, white_pov_delta: int) -> int:
    """Positive = good for ``side``."""
    return white_pov_delta if side == "WHITE" else -white_pov_delta


def _side_label(side: str) -> str:
    return "White" if side == "WHITE" else "Black"


def _benef(side: str) -> str:
    """Beneficiary key for a claim that favors ``side``."""
    return side.lower()


def _benef_opp(side: str) -> str:
    """Beneficiary key for a claim that favors the opponent of ``side``."""
    return "black" if side == "WHITE" else "white"


MAX_CONCESSIONS = 2


def order_claims_for_mover(claims: list[Claim], mover: str) -> list[Claim]:
    """Mover-perspective ordering: the mover's merits first; claims favoring
    the opponent become explicitly tagged concessions, capped at
    ``MAX_CONCESSIONS`` (strongest kept)."""
    mover_key = mover.lower()
    merits = [c for c in claims if c.beneficiary in (mover_key, None)]
    concessions = [c for c in claims if c.beneficiary not in (mover_key, None)]
    concessions = [
        c.model_copy(update={"is_concession": True})
        for c in concessions[:MAX_CONCESSIONS]
    ]
    return merits + concessions


class _Ctx:
    """Everything a rule may look at."""

    def __init__(
        self,
        diff: FeatureDiff,
        phase: str,
        mover: str,
        eval_cp: int | None,
        start_board: chess.Board | None = None,
        leaf_board: chess.Board | None = None,
    ) -> None:
        self.phase = phase
        self.mover = mover  # "WHITE" | "BLACK"
        self.eval_cp = eval_cp
        # Boards behind the diff: rules use them to name squares and files
        # ("passed pawn on e5") — deterministic, no LLM involved.
        self.start_board = start_board
        self.leaf_board = leaf_board
        self.by_name: dict[str, FeatureDelta] = {}
        for d in list(diff.positive) + list(diff.negative):
            self.by_name[d.name] = d

    def color(self, side: str) -> chess.Color:
        return chess.WHITE if side == "WHITE" else chess.BLACK

    def new_squares(self, lookup, side: str) -> list[str]:
        """Squares satisfying `lookup` on the leaf board but not at the start."""
        if self.leaf_board is None:
            return []
        leaf = lookup(self.leaf_board, self.color(side))
        if self.start_board is None:
            return leaf
        start = set(lookup(self.start_board, self.color(side)))
        return [s for s in leaf if s not in start]

    def delta(self, name: str) -> int:
        d = self.by_name.get(name)
        return d.delta_cp if d else 0

    def flag_change(self, name: str) -> str | None:
        d = self.by_name.get(name)
        if d is None or d.flag_before is None or d.flag_after is None:
            return None
        if d.flag_before == d.flag_after:
            return None
        return f"{d.flag_before} -> {d.flag_after}"

    def flag_pair(self, name: str) -> tuple | None:
        d = self.by_name.get(name)
        if d is None or d.flag_before is None or d.flag_after is None:
            return None
        return (d.flag_before, d.flag_after)


Rule = Callable[[_Ctx], list[Claim]]


# ---------------------------------------------------------------------------
# Rules — each returns zero or more claims
# ---------------------------------------------------------------------------


# Material is described in concrete, whole-unit terms (Guid): never fractional
# pawns. piece = bishop/knight only; exchange = rook vs. a minor.
_MAT_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
}
_MAT_VAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
_MAT_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
_NUMWORD = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five"}


def _imbalance(board: chess.Board) -> dict[int, int]:
    """White-minus-Black piece count per type."""
    return {
        pt: len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        for pt in _MAT_ORDER
    }


def _enumerate_extra(extra: dict[int, int]) -> str:
    """e.g. {ROOK:1, BISHOP:1} -> 'a rook and a bishop'; {KNIGHT:2} -> 'two knights'."""
    parts: list[str] = []
    for pt in _MAT_ORDER:
        c = extra.get(pt, 0)
        if c <= 0:
            continue
        name = _MAT_NAME[pt]
        parts.append(f"a {name}" if c == 1 else f"{_NUMWORD.get(c, str(c))} {name}s")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _pawns_phrase(n: int) -> str:
    return "a pawn" if n == 1 else f"{_NUMWORD.get(n, str(n))} pawns"


def describe_material(
    leaf_board: chess.Board, *, changed: bool
) -> tuple[str, str, int] | None:
    """(text, beneficiary, delta_cp) for the material standing, in concrete whole
    units. ``changed`` -> 'has won ...'; else the static 'is ... up' / 'has ...'."""
    imb = _imbalance(leaf_board)
    if all(v == 0 for v in imb.values()):
        return None
    white_extra = {pt: d for pt, d in imb.items() if d > 0}
    black_extra = {pt: -d for pt, d in imb.items() if d < 0}
    val = sum(d * _MAT_VAL[pt] for pt, d in imb.items())
    if val != 0:
        white_subject = val > 0
    else:
        white_subject = sum(white_extra.values()) >= sum(black_extra.values())
    side = "WHITE" if white_subject else "BLACK"
    subj = white_extra if white_subject else black_extra
    opp = black_extra if white_subject else white_extra
    label, benef = _side_label(side), _benef(side)
    delta_cp = max(abs(val) * 100, 100)

    nonpawn_subj = {pt: c for pt, c in subj.items() if pt != chess.PAWN}
    nonpawn_opp = {pt: c for pt, c in opp.items() if pt != chess.PAWN}
    subj_pawns, opp_pawns = subj.get(chess.PAWN, 0), opp.get(chess.PAWN, 0)

    def _single_minor(d: dict[int, int]) -> str | None:
        if sum(d.values()) != 1:
            return None
        if d.get(chess.BISHOP) == 1:
            return "bishop"
        if d.get(chess.KNIGHT) == 1:
            return "knight"
        return None

    # Pure pawn(s).
    if not nonpawn_subj and not nonpawn_opp:
        text = (
            f"{label} has won {_pawns_phrase(subj_pawns)}."
            if changed
            else f"{label} is {_pawns_phrase(subj_pawns)} up."
        )
        return text, benef, max(subj_pawns * 100, 100)

    # Exchange: a rook against a single minor (optionally for pawns).
    if (
        nonpawn_subj.get(chess.ROOK) == 1
        and len(nonpawn_subj) == 1
        and _single_minor(nonpawn_opp)
    ):
        if opp_pawns:
            return (
                f"{label} has won an exchange for {_pawns_phrase(opp_pawns)}.",
                benef,
                delta_cp,
            )
        text = (
            f"{label} has won the exchange."
            if changed
            else f"{label} is up the exchange."
        )
        return text, benef, delta_cp

    # A single minor, opponent has only pawns (or nothing) as compensation.
    minor = _single_minor(nonpawn_subj)
    if minor and not nonpawn_opp:
        if opp_pawns:
            return (
                f"{label} has won a {minor} for {_pawns_phrase(opp_pawns)}.",
                benef,
                delta_cp,
            )
        text = f"{label} has won a {minor}." if changed else f"{label} is up a {minor}."
        return text, benef, delta_cp

    # A single rook, opponent has only pawns.
    if nonpawn_subj.get(chess.ROOK) == 1 and len(nonpawn_subj) == 1 and not nonpawn_opp:
        if opp_pawns:
            return (
                f"{label} has won a rook for {_pawns_phrase(opp_pawns)}.",
                benef,
                delta_cp,
            )
        text = f"{label} has won a rook." if changed else f"{label} is up a rook."
        return text, benef, delta_cp

    # General imbalance: "<subj> for/against <opp>" (against a lone queen).
    subj_desc = _enumerate_extra(subj)
    opp_desc = _enumerate_extra(opp)
    if opp_desc:
        connector = "against" if set(nonpawn_opp) == {chess.QUEEN} else "for"
        return f"{label} has {subj_desc} {connector} {opp_desc}.", benef, delta_cp
    verb = "has won" if changed else "has"
    return f"{label} {verb} {subj_desc}.", benef, delta_cp


def rule_material(ctx: _Ctx) -> list[Claim]:
    """Material standing in concrete, whole-unit terms — the most important
    feature, so it leads the claim list. Fires when the line changes the
    material balance (a capture nets material), describing the actual
    difference / resulting imbalance."""
    if ctx.leaf_board is None:
        return []
    changed = ctx.start_board is None or _imbalance(ctx.start_board) != _imbalance(
        ctx.leaf_board
    )
    if not changed:
        return []
    described = describe_material(ctx.leaf_board, changed=True)
    if described is None:
        return []
    text, benef, delta_cp = described
    return [
        Claim(
            rule_id="material_won",
            beneficiary=benef,
            text=text,
            text_state=text,
            features_involved=["MATERIAL_BALANCE"],
            delta_cp=delta_cp,
        )
    ]


def rule_connected_rooks(ctx: _Ctx) -> list[Claim]:
    """Rooks become connected on the back rank / a file."""
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_ROOKS_CONNECTED"
        pair = ctx.flag_pair(name)
        if (
            pair
            and pair[1] > pair[0]
            and abs(ctx.delta(name)) >= THRESHOLDS["connected_rooks"]
        ):
            out.append(
                Claim(
                    rule_id="rooks_connected",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} connects the rooks.",
                    text_state=f"{_side_label(side)}'s rooks are connected.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_pawn_structure(ctx: _Ctx) -> list[Claim]:
    """The dissertation's pawn-structure rule (§5.4.1), both sides."""
    out: list[Claim] = []
    for side in SIDES:
        feats = [
            f"{side}_PAWN_DOUBLED",
            f"{side}_PAWN_ISOLATED",
            f"{side}_PAWN_BACKWARD",
            f"{side}_PAWN_PASSED",
            f"{side}_PAWN_DUO",
        ]
        total = sum(_toward(side, ctx.delta(f)) for f in feats)
        net = _toward(side, ctx.delta("EVALUATE_PAWNS"))
        if (
            total >= THRESHOLDS["pawn_structure_total"]
            and net >= THRESHOLDS["pawn_structure_evaluate"]
        ):
            out.append(
                Claim(
                    rule_id="pawn_structure_improved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} has improved the pawn structure.",
                    text_state=f"{_side_label(side)}'s pawn structure is now improved.",
                    features_involved=[*feats, "EVALUATE_PAWNS"],
                    delta_cp=total,
                )
            )
        elif (
            -total >= THRESHOLDS["pawn_structure_total"]
            and -net >= THRESHOLDS["pawn_structure_evaluate"]
        ):
            out.append(
                Claim(
                    rule_id="pawn_structure_weakened",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s pawn structure has been weakened.",
                    text_state=f"{_side_label(side)}'s pawn structure is now weaker.",
                    features_involved=[*feats, "EVALUATE_PAWNS"],
                    delta_cp=-total,
                )
            )
    return out


def rule_doubled_pawns(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PAWN_DOUBLED"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            new_files = ctx.new_squares(doubled_pawn_files, side)
            where = f" {new_files[0]}-pawns" if new_files else " pawns"
            out.append(
                Claim(
                    rule_id="doubled_pawns_accepted",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} is left with doubled{where}.",
                    text_state=f"{_side_label(side)} now has doubled{where}.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
        elif pair and pair[1] < pair[0]:
            out.append(
                Claim(
                    rule_id="doubled_pawns_resolved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} gets rid of the doubled pawns.",
                    text_state=f"{_side_label(side)}'s doubled pawns are gone.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_bishop_pair(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_BISHOP_PAIR"
        pair = ctx.flag_pair(name)
        if pair and pair[0] >= 2 and pair[1] < 2:
            out.append(
                Claim(
                    rule_id="bishop_pair_eliminated",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} no longer has the advantage of the bishop pair.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_strong_knight(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_KNIGHTS_OUTPOSTS"
        cent = f"{side}_KNIGHTS_CENTRALIZATION"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["strong_knight"]:
            sqs = ctx.new_squares(outpost_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(
                Claim(
                    rule_id="strong_knight_established",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} establishes a strong knight{where}.",
                    text_state=f"{_side_label(side)} has a strong knight{where}.",
                    features_involved=[name, cent],
                    delta_cp=d,
                    flag_note=ctx.flag_change(name),
                )
            )
        elif -d >= THRESHOLDS["strong_knight"]:
            out.append(
                Claim(
                    rule_id="strong_knight_lost",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} no longer has a strong knight.",
                    features_involved=[name, cent],
                    delta_cp=-d,
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_bad_bishop(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_BAD_BISHOP"
        pair = ctx.flag_pair(name)
        # Require a meaningful swing — the flag flickers as pawns leave the
        # bishop's colour along best play, which over-reported "solved".
        if pair and abs(ctx.delta(name)) < THRESHOLDS["bad_bishop"]:
            continue
        if pair and pair[1] > pair[0]:
            sqs = ctx.new_squares(bad_bishop_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(
                Claim(
                    rule_id="bad_bishop_created",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)} is left with a bad bishop{where}.",
                    text_state=f"{_side_label(side)}'s bishop{where} is now bad.",
                    features_involved=[
                        name,
                        f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR",
                        f"{side}_BISHOPS_MOBILITY",
                    ],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
        elif pair and pair[1] < pair[0]:
            start_sqs = (
                bad_bishop_squares(ctx.start_board, ctx.color(side))
                if ctx.start_board is not None
                else []
            )
            where = f" on {start_sqs[0]}" if start_sqs else ""
            out.append(
                Claim(
                    rule_id="bad_bishop_solved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} solves the problem of the bad bishop{where}.",
                    text_state=f"{_side_label(side)}'s bad bishop{where} is no longer a problem.",
                    features_involved=[
                        name,
                        f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR",
                        f"{side}_BISHOPS_MOBILITY",
                    ],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_rook_activity(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        seventh = f"{side}_ROOK_ON_SEVENTH"
        pair7 = ctx.flag_pair(seventh)
        if pair7 and pair7[1] > pair7[0]:
            sqs = ctx.new_squares(rooks_on_seventh_squares, side)
            where = f" ({sqs[0]})" if sqs else ""
            out.append(
                Claim(
                    rule_id="rook_reaches_seventh",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s rook reaches the seventh rank{where}.",
                    text_state=f"{_side_label(side)} has a rook on the seventh rank{where}.",
                    features_involved=[seventh],
                    delta_cp=abs(ctx.delta(seventh)),
                    flag_note=ctx.flag_change(seventh),
                )
            )
            continue
        feats = [f"{side}_ROOK_OPEN_FILE", f"{side}_ROOK_HALF_OPEN_FILE"]
        d = sum(_toward(side, ctx.delta(f)) for f in feats)
        if d >= THRESHOLDS["rook_activity"]:
            out.append(
                Claim(
                    rule_id="rooks_activated",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s rooks become more active on the open files.",
                    text_state=f"{_side_label(side)}'s rooks are active on the open files.",
                    features_involved=feats,
                    delta_cp=d,
                )
            )
    return out


def rule_rook_behind_passer(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_ROOK_BEHIND_PASSED_PAWN"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            out.append(
                Claim(
                    rule_id="rook_behind_passer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s rook gets behind the passed pawn.",
                    text_state=f"{_side_label(side)}'s rook is behind the passed pawn.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_passed_pawn(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PAWN_PASSED"
        pair = ctx.flag_pair(name)
        d = _toward(side, ctx.delta(name))
        if pair and pair[1] > pair[0]:
            sqs = ctx.new_squares(passed_pawn_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(
                Claim(
                    rule_id="passed_pawn_created",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} obtains a passed pawn{where}.",
                    text_state=f"{_side_label(side)} has a passed pawn{where}.",
                    features_involved=[name],
                    delta_cp=max(d, 0),
                    flag_note=ctx.flag_change(name),
                )
            )
        elif pair and pair[1] == pair[0] and pair[1] > 0 and d >= 15:
            out.append(
                Claim(
                    rule_id="passed_pawn_advances",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s passed pawn advances dangerously.",
                    text_state=f"{_side_label(side)}'s passed pawn is far advanced.",
                    features_involved=[name],
                    delta_cp=d,
                )
            )
    return out


def rule_king_safety(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        feats = [
            f"{side}_KING_SHIELD",
            f"{side}_KING_ZONE_ATTACKERS",
            f"{side}_BACK_RANK",
        ]
        d = sum(_toward(side, ctx.delta(f)) for f in feats)
        opp = "BLACK" if side == "WHITE" else "WHITE"
        opp_tropism = _toward(opp, ctx.delta(f"{opp}_KING_TROPISM"))
        if (
            -d >= THRESHOLDS["king_safety"]
            and opp_tropism >= THRESHOLDS["king_tropism_corroborate"]
        ):
            out.append(
                Claim(
                    rule_id="king_under_pressure",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s king comes under pressure.",
                    text_state=f"{_side_label(side)}'s king is under pressure.",
                    features_involved=[*feats, f"{opp}_KING_TROPISM"],
                    delta_cp=-d,
                )
            )
        elif (
            d >= THRESHOLDS["king_safety"]
            and -opp_tropism >= THRESHOLDS["king_tropism_corroborate"]
        ):
            out.append(
                Claim(
                    rule_id="king_safer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s king is now safer.",
                    text_state=f"{_side_label(side)}'s king is safe.",
                    features_involved=[*feats, f"{opp}_KING_TROPISM"],
                    delta_cp=d,
                )
            )
    return out


def rule_back_rank(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_BACK_RANK"
        pair = ctx.flag_pair(name)
        if pair and pair[0] == 0 and pair[1] == 1:
            out.append(
                Claim(
                    rule_id="back_rank_weakness",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s back rank becomes vulnerable.",
                    text_state=f"{_side_label(side)}'s back rank is vulnerable.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_piece_activity(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    # When material changed, the mobility swing is mostly a side effect of the
    # capture, not a genuine activity gain — let rule_material speak instead.
    if abs(ctx.delta("MATERIAL_BALANCE")) >= 100:
        return out
    for side in SIDES:
        name = f"{side}_PIECE_ACTIVITY"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["piece_activity"]:
            out.append(
                Claim(
                    rule_id="activity_improved",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} has improved the activity of the pieces.",
                    text_state=f"{_side_label(side)}'s pieces are actively placed.",
                    features_involved=[name],
                    delta_cp=d,
                )
            )
        elif -d >= THRESHOLDS["piece_activity"]:
            out.append(
                Claim(
                    rule_id="activity_reduced",
                    beneficiary=_benef_opp(side),
                    text=f"{_side_label(side)}'s pieces become more passive.",
                    text_state=f"{_side_label(side)}'s pieces are passive.",
                    features_involved=[name],
                    delta_cp=-d,
                )
            )
    return out


def rule_center_and_space(ctx: _Ctx) -> list[Claim]:
    out: list[Claim] = []
    for side in SIDES:
        c = _toward(side, ctx.delta(f"{side}_CENTER_CONTROL"))
        s = _toward(side, ctx.delta(f"{side}_SPACE"))
        if c >= THRESHOLDS["center_control"]:
            out.append(
                Claim(
                    rule_id="center_control",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} takes control of the center.",
                    text_state=f"{_side_label(side)} controls the center.",
                    features_involved=[f"{side}_CENTER_CONTROL"],
                    delta_cp=c,
                )
            )
        # Independent of center — center no longer masks a real space gain.
        if s >= THRESHOLDS["space"]:
            out.append(
                Claim(
                    rule_id="space_gained",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} gains space.",
                    text_state=f"{_side_label(side)} has a space advantage.",
                    features_involved=[f"{side}_SPACE"],
                    delta_cp=s,
                )
            )
    return out


# --- endgame pack ---


def rule_king_activity(ctx: _Ctx) -> list[Claim]:
    if ctx.phase != "end":
        return []
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_KING_ACTIVITY"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["king_activity_endgame"]:
            out.append(
                Claim(
                    rule_id="king_activated",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s king becomes active.",
                    text_state=f"{_side_label(side)}'s king is active.",
                    features_involved=[name],
                    delta_cp=d,
                )
            )
    return out


def rule_outside_passer(ctx: _Ctx) -> list[Claim]:
    if ctx.phase != "end":
        return []
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_OUTSIDE_PASSER"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            out.append(
                Claim(
                    rule_id="outside_passer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)} obtains an outside passed pawn.",
                    text_state=f"{_side_label(side)} has an outside passed pawn.",
                    features_involved=[name],
                    delta_cp=abs(ctx.delta(name)),
                    flag_note=ctx.flag_change(name),
                )
            )
    return out


def rule_passer_escort(ctx: _Ctx) -> list[Claim]:
    if ctx.phase != "end":
        return []
    out: list[Claim] = []
    for side in SIDES:
        name = f"{side}_PASSER_KING_ESCORT"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["passer_escort"]:
            out.append(
                Claim(
                    rule_id="king_escorts_passer",
                    beneficiary=_benef(side),
                    text=f"{_side_label(side)}'s king escorts the passed pawn forward.",
                    text_state=f"{_side_label(side)}'s king supports the passed pawn.",
                    features_involved=[name, f"{side}_KING_ACTIVITY"],
                    delta_cp=d,
                )
            )
    return out


ALL_RULES: list[Rule] = [
    rule_material,
    rule_pawn_structure,
    rule_doubled_pawns,
    rule_bishop_pair,
    rule_strong_knight,
    rule_bad_bishop,
    rule_rook_activity,
    rule_rook_behind_passer,
    rule_passed_pawn,
    rule_king_safety,
    rule_back_rank,
    rule_piece_activity,
    rule_center_and_space,
    rule_connected_rooks,
    rule_king_activity,
    rule_outside_passer,
    rule_passer_escort,
]

MAX_CLAIMS_PER_MOVE = 4


def run_rules(
    diff: FeatureDiff,
    *,
    phase: str,
    mover: str,
    eval_cp: int | None = None,
    start_board: chess.Board | None = None,
    leaf_board: chess.Board | None = None,
) -> list[Claim]:
    """Fire all rules; keep the strongest few claims."""
    ctx = _Ctx(
        diff, phase, mover, eval_cp, start_board=start_board, leaf_board=leaf_board
    )
    claims: list[Claim] = []
    for rule in ALL_RULES:
        try:
            claims.extend(rule(ctx))
        except Exception as e:
            logger.warning("rule %s failed: %s", getattr(rule, "__name__", rule), e)
    # Every claim must clear the magnitude floor — a flag flip alone (e.g. a
    # routine bishop trade) is no longer enough to earn a sentence.
    claims = [c for c in claims if c.delta_cp >= THRESHOLDS["min_claim_cp"]]
    claims.sort(key=lambda c: -c.delta_cp)
    return claims[:MAX_CLAIMS_PER_MOVE]


# ---------------------------------------------------------------------------
# Verdict (qualitative assessment of the eval)
# ---------------------------------------------------------------------------


def _adv_bracket(cp: int) -> str | None:
    """Advantage label for |cp| >= 25, else None (equality)."""
    mag = abs(cp)
    if mag < 25:
        return None
    if mag < 75:
        return "a slight advantage"
    if mag < 150:
        return "a clear advantage"
    if mag < 300:
        return "a winning advantage"
    return "a decisive advantage"


def verdict_for_transition(
    before_cp: int | None,
    after_cp: int | None,
    eval_mate: int | None,
    mover: str,
) -> str:
    """Verdict phrased as the eval *story* of the move from the mover's seat:
    held / extended / slipped / thrown away / conceded — far more informative
    than the absolute bracket alone. Falls back to ``verdict_for_eval``."""
    if eval_mate is not None:
        side = "White" if eval_mate > 0 else "Black"
        return f"leads to forced mate for {side}"
    if before_cp is None or after_cp is None:
        return verdict_for_eval(after_cp, eval_mate)

    opp = "Black" if mover == "White" else "White"
    sign = 1 if mover == "White" else -1
    b = sign * int(before_cp)  # mover POV
    a = sign * int(after_cp)
    d = a - b
    a_lab = _adv_bracket(a)
    _adv_bracket(b)

    if abs(d) < 35:  # nothing really changed
        if a_lab is None:
            return "holds the balance"
        if a > 0:
            return f"maintains {mover}'s advantage"
        return verdict_for_eval(after_cp)

    if d < 0:  # the mover lost ground
        if b > 25 and a > 25:
            return f"lets {mover}'s advantage shrink"
        if b > 25 and -25 <= a <= 25:
            return "lets the advantage slip away to equality"
        if b > 25 and a < -25:
            return f"throws away the advantage and hands {opp} {_adv_bracket(-a)}"
        if -25 <= b <= 25 and a < -25:
            return f"concedes {_adv_bracket(-a)} to {opp}"
        if b < -25 and a < b:
            return f"makes matters worse — {opp} now has {_adv_bracket(-a)}"
        return verdict_for_eval(after_cp)

    # the mover gained ground (usually cashing in on the opponent's error)
    if a > 25 and b <= 25:
        return f"seizes {a_lab} for {mover}"
    if a > 25 and b > 25:
        return f"extends {mover}'s advantage"
    if -25 <= a <= 25:
        return "restores the balance"
    return f"fights back, though {opp} keeps {_adv_bracket(-a)}"


def verdict_for_eval(eval_cp: int | None, eval_mate: int | None = None) -> str:
    if eval_mate is not None:
        side = "White" if eval_mate > 0 else "Black"
        return f"leads to forced mate for {side}"
    if eval_cp is None:
        return ""
    cp = int(eval_cp)
    mag = abs(cp)
    side = "White" if cp > 0 else "Black"
    if mag < 25:
        return "leads to equality"
    if mag < 75:
        return f"leads to a slight advantage for {side}"
    if mag < 150:
        return f"leads to a clear advantage for {side}"
    if mag < 300:
        return f"gives {side} a winning advantage"
    return f"leaves {side} with a decisive advantage"


# ---------------------------------------------------------------------------
# CommentFacts builder
# ---------------------------------------------------------------------------

_MATE_SCORE = 1000000


def _decode_eval(cp: int | None) -> tuple:
    """Split a mate-encoded engine score into (cp, mate)."""
    if cp is None:
        return None, None
    if abs(cp) > _MATE_SCORE - 1000:
        n = _MATE_SCORE - abs(cp)
        return None, (n if cp > 0 else -n)
    return int(cp), None


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


def build_comment_facts(
    row: AnalyzedMoveData,
    me: MoveEvent,
    *,
    depth: int = 16,
) -> CommentFacts | None:
    """Assemble the move's inviolable facts from data the engine pass already paid for."""
    phase_raw = row.phase_raw or "mid"
    if phase_raw == "early":
        return None

    board_before = chess.Board(me.fen_before)
    mover = "White" if board_before.turn == chess.WHITE else "Black"

    eval_cp, eval_mate = _decode_eval(me.eval_after_cp)
    eval_before_cp, _before_mate = _decode_eval(me.eval_before_cp)

    eng = (row.hidden_features or {}).get("_engine") or {}
    after_pv = list(eng.get("after_pv_uci") or [])

    played_line = envisioned_for_played_move(
        me.fen_before,
        me.uci,
        after_pv,
        played_eval_cp=me.eval_after_cp,
        depth=depth,
    )
    leaf_board = chess.Board(played_line.leaf_fen)
    start_vec = compute_feature_vector_fen(me.fen_before)
    leaf_vec = compute_feature_vector_fen(played_line.leaf_fen)
    diff = diff_vectors(start_vec, leaf_vec)
    claims = run_rules(
        diff,
        phase=phase_raw,
        mover=mover.upper(),
        eval_cp=me.eval_after_cp,
        start_board=board_before,
        leaf_board=leaf_board,
    )
    claims = order_claims_for_mover(claims, mover)
    # For dubious moves, opponent-favoring claims are not trade-offs — they ARE
    # the explanation of the eval swing.
    mq = me.move_quality.value if me.move_quality else ""
    concession_mode = (
        "consequence" if mq in ("inaccuracy", "mistake", "blunder") else "tradeoff"
    )

    # The opponent's punishing reply (board-level "why it is bad"). Only for
    # real mistakes — calling a routine recapture a "punishment" reads wrong.
    refutation_san: str | None = None
    if mq in ("mistake", "blunder") and len(played_line.line_san) >= 2:
        refutation_san = played_line.line_san[1]

    # At decisive evals positional claims are noise: a passed pawn does not
    # matter in mate-in-4. Keep the verdict + refutation only.
    if eval_mate is not None or (eval_cp is not None and abs(eval_cp) > 500):
        claims = []

    # Fallback explanation when no positional rule fired — especially on
    # tactical mistakes the positional rules don't model. Derive ONE grounded
    # claim from material / the refutation / the engine's preference so the
    # comment always says WHY, not just the verdict. All facts (material delta,
    # best-move SAN), so the contract holds.
    if not claims and mq in ("inaccuracy", "mistake", "blunder"):
        opp = "black" if mover == "White" else "white"
        mover_sign = 1 if mover == "White" else -1

        def _net_material(vec) -> int:
            fv = vec.get("MATERIAL_BALANCE")
            return fv.value_cp if fv is not None else 0

        mat_delta = mover_sign * (_net_material(leaf_vec) - _net_material(start_vec))
        best_san = me.best_move_san
        fb: Claim | None = None
        if mat_delta <= -100:
            txt = f"{mover} loses material"
            if best_san:
                txt += f"; {best_san} held the balance"
            fb = Claim(
                rule_id="tactical_material_loss",
                text=txt + ".",
                beneficiary=opp,
                delta_cp=abs(mat_delta),
                features_involved=["MATERIAL_BALANCE"],
            )
        elif refutation_san is None and best_san:
            # No material swing and no refutation sentence from the template:
            # point at the engine's preference so the reasons list is not empty.
            fb = Claim(
                rule_id="eval_concession",
                text=f"the engine preferred {best_san} here.",
                beneficiary=opp,
                delta_cp=60,
            )
        if fb is not None:
            claims = [fb]

    # Better alternative (Guid's option 3): only when the played move measurably
    # loses ground against the engine's preference.
    better: BestAlternative | None = None
    if (
        me.best_move_uci
        and me.best_move_uci != me.uci
        and me.best_move_eval_cp is not None
        and me.eval_after_cp is not None
    ):
        gap = me.best_move_eval_cp - me.eval_after_cp
        gap_for_mover = gap if mover == "White" else -gap
        # A ?/?? move must always show what was better (Guid) — bypass the gap
        # gate for mistakes/blunders (the decisive-position filter already kept
        # only the ones that are real mistakes).
        force_alt = mq in ("mistake", "blunder")
        if force_alt or gap_for_mover >= THRESHOLDS["better_alternative_gap"]:
            best_pv_uci: list[str] = []
            if row.pvs and row.pvs[0]:
                best_pv_uci = [
                    str(m.move) for m in row.pvs[0] if getattr(m, "move", None)
                ]
            best_cp, best_mate = _decode_eval(me.best_move_eval_cp)
            best_line = envisioned_for_best_move(
                me.fen_before,
                best_pv_uci,
                best_eval_cp=best_cp,
                depth=depth,
            )
            best_leaf_vec = compute_feature_vector_fen(best_line.leaf_fen)
            best_diff = diff_vectors(start_vec, best_leaf_vec)
            best_claims = run_rules(
                best_diff,
                phase=phase_raw,
                mover=mover.upper(),
                eval_cp=best_cp,
                start_board=board_before,
                leaf_board=chess.Board(best_line.leaf_fen),
            )
            # The alternative is the move the mover SHOULD have played: show
            # only its merits (max 2) and never repeat the main block's claims.
            main_texts = {c.text for c in claims}
            mover_key = mover.lower()
            best_claims = [
                c
                for c in best_claims
                if c.beneficiary in (mover_key, None) and c.text not in main_texts
            ][:2]
            better = BestAlternative(
                san=me.best_move_san or me.best_move_uci,
                uci=me.best_move_uci,
                eval_cp=best_cp,
                verdict=verdict_for_eval(best_cp, best_mate),
                display_line=best_line,
                claims=best_claims,
            )

    # Feature progression along the displayed lines: emit every feature so the
    # navigator can chart the game-vs-line curve for all of them, not just the
    # fired-rule ones (the per-point vector is computed in full regardless).
    played_line = played_line.model_copy(
        update={"feature_series": _line_feature_series(me.fen_before, played_line.fens)}
    )
    if better is not None and better.display_line is not None:
        better = better.model_copy(
            update={
                "display_line": better.display_line.model_copy(
                    update={
                        "feature_series": _line_feature_series(
                            me.fen_before, better.display_line.fens
                        )
                    }
                )
            }
        )

    return CommentFacts(
        ply=me.ply,
        san=me.san,
        uci=me.uci,
        mover=mover,
        phase=phase_raw,
        verdict=verdict_for_transition(eval_before_cp, eval_cp, eval_mate, mover),
        eval_cp=eval_cp,
        eval_before_cp=eval_before_cp,
        eval_mate=eval_mate,
        refutation_san=refutation_san,
        depth=depth,
        engine="Stockfish",
        display_line=played_line,
        claims=claims,
        feature_diff=diff,
        better_alternative=better,
        concession_mode=concession_mode,
    )
