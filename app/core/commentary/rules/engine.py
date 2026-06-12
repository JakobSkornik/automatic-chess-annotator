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
from typing import Callable, Dict, List, Optional

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
    EnvisionedLine,
    FeatureDelta,
    FeatureDiff,
)

logger = logging.getLogger(__name__)

THRESHOLDS: Dict[str, int] = {
    "pawn_structure_total": 14,     # Z in the dissertation's pawn rule
    "pawn_structure_evaluate": 8,   # Y — net EVALUATE_PAWNS shift
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
    "min_claim_cp": 8,              # ignore fired rules weaker than this
    "better_alternative_gap": 50,   # cp loss before the best move is shown
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


def order_claims_for_mover(claims: List[Claim], mover: str) -> List[Claim]:
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
        eval_cp: Optional[int],
        start_board: Optional[chess.Board] = None,
        leaf_board: Optional[chess.Board] = None,
    ) -> None:
        self.phase = phase
        self.mover = mover  # "WHITE" | "BLACK"
        self.eval_cp = eval_cp
        # Boards behind the diff: rules use them to name squares and files
        # ("passed pawn on e5") — deterministic, no LLM involved.
        self.start_board = start_board
        self.leaf_board = leaf_board
        self.by_name: Dict[str, FeatureDelta] = {}
        for d in list(diff.positive) + list(diff.negative):
            self.by_name[d.name] = d

    def color(self, side: str) -> chess.Color:
        return chess.WHITE if side == "WHITE" else chess.BLACK

    def new_squares(self, lookup, side: str) -> List[str]:
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

    def flag_change(self, name: str) -> Optional[str]:
        d = self.by_name.get(name)
        if d is None or d.flag_before is None or d.flag_after is None:
            return None
        if d.flag_before == d.flag_after:
            return None
        return f"{d.flag_before} -> {d.flag_after}"

    def flag_pair(self, name: str) -> Optional[tuple]:
        d = self.by_name.get(name)
        if d is None or d.flag_before is None or d.flag_after is None:
            return None
        return (d.flag_before, d.flag_after)


Rule = Callable[[_Ctx], List[Claim]]


# ---------------------------------------------------------------------------
# Rules — each returns zero or more claims
# ---------------------------------------------------------------------------

def rule_pawn_structure(ctx: _Ctx) -> List[Claim]:
    """The dissertation's pawn-structure rule (§5.4.1), both sides."""
    out: List[Claim] = []
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
        if total >= THRESHOLDS["pawn_structure_total"] and net >= THRESHOLDS["pawn_structure_evaluate"]:
            out.append(Claim(
                rule_id="pawn_structure_improved",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} has improved the pawn structure.",
                text_state=f"{_side_label(side)}'s pawn structure is now improved.",
                features_involved=feats + ["EVALUATE_PAWNS"],
                delta_cp=total,
            ))
        elif -total >= THRESHOLDS["pawn_structure_total"] and -net >= THRESHOLDS["pawn_structure_evaluate"]:
            out.append(Claim(
                rule_id="pawn_structure_weakened",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)}'s pawn structure has been weakened.",
                text_state=f"{_side_label(side)}'s pawn structure is now weaker.",
                features_involved=feats + ["EVALUATE_PAWNS"],
                delta_cp=-total,
            ))
    return out


def rule_doubled_pawns(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_PAWN_DOUBLED"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            new_files = ctx.new_squares(doubled_pawn_files, side)
            where = f" {new_files[0]}-pawns" if new_files else " pawns"
            out.append(Claim(
                rule_id="doubled_pawns_accepted",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)} is left with doubled{where}.",
                text_state=f"{_side_label(side)} now has doubled{where}.",
                features_involved=[name],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
        elif pair and pair[1] < pair[0]:
            out.append(Claim(
                rule_id="doubled_pawns_resolved",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} gets rid of the doubled pawns.",
                text_state=f"{_side_label(side)}'s doubled pawns are gone.",
                features_involved=[name],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
    return out


def rule_bishop_pair(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_BISHOP_PAIR"
        pair = ctx.flag_pair(name)
        if pair and pair[0] >= 2 and pair[1] < 2:
            out.append(Claim(
                rule_id="bishop_pair_eliminated",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)} no longer has the advantage of the bishop pair.",
                features_involved=[name],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
    return out


def rule_strong_knight(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_KNIGHTS_OUTPOSTS"
        cent = f"{side}_KNIGHTS_CENTRALIZATION"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["strong_knight"]:
            sqs = ctx.new_squares(outpost_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(Claim(
                rule_id="strong_knight_established",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} establishes a strong knight{where}.",
                text_state=f"{_side_label(side)} has a strong knight{where}.",
                features_involved=[name, cent],
                delta_cp=d,
                flag_note=ctx.flag_change(name),
            ))
        elif -d >= THRESHOLDS["strong_knight"]:
            out.append(Claim(
                rule_id="strong_knight_lost",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)} no longer has a strong knight.",
                features_involved=[name, cent],
                delta_cp=-d,
                flag_note=ctx.flag_change(name),
            ))
    return out


def rule_bad_bishop(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_BAD_BISHOP"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            sqs = ctx.new_squares(bad_bishop_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(Claim(
                rule_id="bad_bishop_created",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)} is left with a bad bishop{where}.",
                text_state=f"{_side_label(side)}'s bishop{where} is now bad.",
                features_involved=[name, f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR", f"{side}_BISHOPS_MOBILITY"],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
        elif pair and pair[1] < pair[0]:
            start_sqs = (
                bad_bishop_squares(ctx.start_board, ctx.color(side))
                if ctx.start_board is not None
                else []
            )
            where = f" on {start_sqs[0]}" if start_sqs else ""
            out.append(Claim(
                rule_id="bad_bishop_solved",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} solves the problem of the bad bishop{where}.",
                text_state=f"{_side_label(side)}'s bad bishop{where} is no longer a problem.",
                features_involved=[name, f"{side}_BISHOP_PLUS_PAWNS_ON_COLOR", f"{side}_BISHOPS_MOBILITY"],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
    return out


def rule_rook_activity(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        seventh = f"{side}_ROOK_ON_SEVENTH"
        pair7 = ctx.flag_pair(seventh)
        if pair7 and pair7[1] > pair7[0]:
            sqs = ctx.new_squares(rooks_on_seventh_squares, side)
            where = f" ({sqs[0]})" if sqs else ""
            out.append(Claim(
                rule_id="rook_reaches_seventh",
                beneficiary=_benef(side),
                text=f"{_side_label(side)}'s rook reaches the seventh rank{where}.",
                text_state=f"{_side_label(side)} has a rook on the seventh rank{where}.",
                features_involved=[seventh],
                delta_cp=abs(ctx.delta(seventh)),
                flag_note=ctx.flag_change(seventh),
            ))
            continue
        feats = [f"{side}_ROOK_OPEN_FILE", f"{side}_ROOK_HALF_OPEN_FILE"]
        d = sum(_toward(side, ctx.delta(f)) for f in feats)
        if d >= THRESHOLDS["rook_activity"]:
            out.append(Claim(
                rule_id="rooks_activated",
                beneficiary=_benef(side),
                text=f"{_side_label(side)}'s rooks become more active on the open files.",
                text_state=f"{_side_label(side)}'s rooks are active on the open files.",
                features_involved=feats,
                delta_cp=d,
            ))
    return out


def rule_rook_behind_passer(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_ROOK_BEHIND_PASSED_PAWN"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            out.append(Claim(
                rule_id="rook_behind_passer",
                beneficiary=_benef(side),
                text=f"{_side_label(side)}'s rook gets behind the passed pawn.",
                text_state=f"{_side_label(side)}'s rook is behind the passed pawn.",
                features_involved=[name],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
    return out


def rule_passed_pawn(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_PAWN_PASSED"
        pair = ctx.flag_pair(name)
        d = _toward(side, ctx.delta(name))
        if pair and pair[1] > pair[0]:
            sqs = ctx.new_squares(passed_pawn_squares, side)
            where = f" on {sqs[0]}" if sqs else ""
            out.append(Claim(
                rule_id="passed_pawn_created",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} obtains a passed pawn{where}.",
                text_state=f"{_side_label(side)} has a passed pawn{where}.",
                features_involved=[name],
                delta_cp=max(d, 0),
                flag_note=ctx.flag_change(name),
            ))
        elif pair and pair[1] == pair[0] and pair[1] > 0 and d >= 15:
            out.append(Claim(
                rule_id="passed_pawn_advances",
                beneficiary=_benef(side),
                text=f"{_side_label(side)}'s passed pawn advances dangerously.",
                text_state=f"{_side_label(side)}'s passed pawn is far advanced.",
                features_involved=[name],
                delta_cp=d,
            ))
    return out


def rule_king_safety(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        feats = [f"{side}_KING_SHIELD", f"{side}_KING_ZONE_ATTACKERS", f"{side}_BACK_RANK"]
        d = sum(_toward(side, ctx.delta(f)) for f in feats)
        opp = "BLACK" if side == "WHITE" else "WHITE"
        opp_tropism = _toward(opp, ctx.delta(f"{opp}_KING_TROPISM"))
        if -d >= THRESHOLDS["king_safety"] and opp_tropism >= THRESHOLDS["king_tropism_corroborate"]:
            out.append(Claim(
                rule_id="king_under_pressure",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)}'s king comes under pressure.",
                text_state=f"{_side_label(side)}'s king is under pressure.",
                features_involved=feats + [f"{opp}_KING_TROPISM"],
                delta_cp=-d,
            ))
        elif d >= THRESHOLDS["king_safety"] and -opp_tropism >= THRESHOLDS["king_tropism_corroborate"]:
            out.append(Claim(
                rule_id="king_safer",
                beneficiary=_benef(side),
                text=f"{_side_label(side)}'s king is now safer.",
                text_state=f"{_side_label(side)}'s king is safe.",
                features_involved=feats + [f"{opp}_KING_TROPISM"],
                delta_cp=d,
            ))
    return out


def rule_back_rank(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_BACK_RANK"
        pair = ctx.flag_pair(name)
        if pair and pair[0] == 0 and pair[1] == 1:
            out.append(Claim(
                rule_id="back_rank_weakness",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)}'s back rank becomes vulnerable.",
                text_state=f"{_side_label(side)}'s back rank is vulnerable.",
                features_involved=[name],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
    return out


def rule_piece_activity(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_PIECE_ACTIVITY"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["piece_activity"]:
            out.append(Claim(
                rule_id="activity_improved",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} has improved the activity of the pieces.",
                text_state=f"{_side_label(side)}'s pieces are actively placed.",
                features_involved=[name],
                delta_cp=d,
            ))
        elif -d >= THRESHOLDS["piece_activity"]:
            out.append(Claim(
                rule_id="activity_reduced",
                beneficiary=_benef_opp(side),
                text=f"{_side_label(side)}'s pieces become more passive.",
                text_state=f"{_side_label(side)}'s pieces are passive.",
                features_involved=[name],
                delta_cp=-d,
            ))
    return out


def rule_center_and_space(ctx: _Ctx) -> List[Claim]:
    out: List[Claim] = []
    for side in SIDES:
        c = _toward(side, ctx.delta(f"{side}_CENTER_CONTROL"))
        s = _toward(side, ctx.delta(f"{side}_SPACE"))
        if c >= THRESHOLDS["center_control"]:
            out.append(Claim(
                rule_id="center_control",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} takes control of the center.",
                text_state=f"{_side_label(side)} controls the center.",
                features_involved=[f"{side}_CENTER_CONTROL"],
                delta_cp=c,
            ))
        elif s >= THRESHOLDS["space"]:
            out.append(Claim(
                rule_id="space_gained",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} gains space.",
                text_state=f"{_side_label(side)} has a space advantage.",
                features_involved=[f"{side}_SPACE"],
                delta_cp=s,
            ))
    return out


# --- endgame pack ---

def rule_king_activity(ctx: _Ctx) -> List[Claim]:
    if ctx.phase != "end":
        return []
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_KING_ACTIVITY"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["king_activity_endgame"]:
            out.append(Claim(
                rule_id="king_activated",
                beneficiary=_benef(side),
                text=f"{_side_label(side)}'s king becomes active.",
                text_state=f"{_side_label(side)}'s king is active.",
                features_involved=[name],
                delta_cp=d,
            ))
    return out


def rule_outside_passer(ctx: _Ctx) -> List[Claim]:
    if ctx.phase != "end":
        return []
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_OUTSIDE_PASSER"
        pair = ctx.flag_pair(name)
        if pair and pair[1] > pair[0]:
            out.append(Claim(
                rule_id="outside_passer",
                beneficiary=_benef(side),
                text=f"{_side_label(side)} obtains an outside passed pawn.",
                text_state=f"{_side_label(side)} has an outside passed pawn.",
                features_involved=[name],
                delta_cp=abs(ctx.delta(name)),
                flag_note=ctx.flag_change(name),
            ))
    return out


def rule_passer_escort(ctx: _Ctx) -> List[Claim]:
    if ctx.phase != "end":
        return []
    out: List[Claim] = []
    for side in SIDES:
        name = f"{side}_PASSER_KING_ESCORT"
        d = _toward(side, ctx.delta(name))
        if d >= THRESHOLDS["passer_escort"]:
            out.append(Claim(
                rule_id="king_escorts_passer",
                beneficiary=_benef(side),
                text=f"{_side_label(side)}'s king escorts the passed pawn forward.",
                text_state=f"{_side_label(side)}'s king supports the passed pawn.",
                features_involved=[name, f"{side}_KING_ACTIVITY"],
                delta_cp=d,
            ))
    return out


ALL_RULES: List[Rule] = [
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
    eval_cp: Optional[int] = None,
    start_board: Optional[chess.Board] = None,
    leaf_board: Optional[chess.Board] = None,
) -> List[Claim]:
    """Fire all rules; keep the strongest few claims."""
    ctx = _Ctx(diff, phase, mover, eval_cp, start_board=start_board, leaf_board=leaf_board)
    claims: List[Claim] = []
    for rule in ALL_RULES:
        try:
            claims.extend(rule(ctx))
        except Exception as e:
            logger.warning("rule %s failed: %s", getattr(rule, "__name__", rule), e)
    claims = [c for c in claims if c.delta_cp >= THRESHOLDS["min_claim_cp"] or c.flag_note]
    claims.sort(key=lambda c: -c.delta_cp)
    return claims[:MAX_CLAIMS_PER_MOVE]


# ---------------------------------------------------------------------------
# Verdict (qualitative assessment of the eval)
# ---------------------------------------------------------------------------

def _adv_bracket(cp: int) -> Optional[str]:
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
    before_cp: Optional[int],
    after_cp: Optional[int],
    eval_mate: Optional[int],
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
    b_lab = _adv_bracket(b)

    if abs(d) < 35:  # nothing really changed
        if a_lab is None:
            return "holds the balance"
        if a > 0:
            return f"maintains {mover}'s advantage"
        return verdict_for_eval(after_cp)

    if d < 0:  # the mover lost ground
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


def verdict_for_eval(eval_cp: Optional[int], eval_mate: Optional[int] = None) -> str:
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


def _decode_eval(cp: Optional[int]) -> tuple:
    """Split a mate-encoded engine score into (cp, mate)."""
    if cp is None:
        return None, None
    if abs(cp) > _MATE_SCORE - 1000:
        n = _MATE_SCORE - abs(cp)
        return None, (n if cp > 0 else -n)
    return int(cp), None


def build_comment_facts(
    row: AnalyzedMoveData,
    me: MoveEvent,
    *,
    depth: int = 16,
) -> Optional[CommentFacts]:
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
    concession_mode = "consequence" if mq in ("inaccuracy", "mistake", "blunder") else "tradeoff"

    # The opponent's punishing reply (board-level "why it is bad"). Only for
    # real mistakes — calling a routine recapture a "punishment" reads wrong.
    refutation_san: Optional[str] = None
    if mq in ("mistake", "blunder") and len(played_line.line_san) >= 2:
        refutation_san = played_line.line_san[1]

    # At decisive evals positional claims are noise: a passed pawn does not
    # matter in mate-in-4. Keep the verdict + refutation only.
    if eval_mate is not None or (eval_cp is not None and abs(eval_cp) > 500):
        claims = []

    # Better alternative (Guid's option 3): only when the played move measurably
    # loses ground against the engine's preference.
    better: Optional[BestAlternative] = None
    if (
        me.best_move_uci
        and me.best_move_uci != me.uci
        and me.best_move_eval_cp is not None
        and me.eval_after_cp is not None
    ):
        gap = me.best_move_eval_cp - me.eval_after_cp
        gap_for_mover = gap if mover == "White" else -gap
        if gap_for_mover >= THRESHOLDS["better_alternative_gap"]:
            best_pv_uci: List[str] = []
            if row.pvs and row.pvs[0]:
                best_pv_uci = [str(m.move) for m in row.pvs[0] if getattr(m, "move", None)]
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
                c for c in best_claims
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
