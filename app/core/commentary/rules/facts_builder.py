"""CommentFacts assembly: the played line, its claims, the better alternative,
realization tagging, and the inviolable per-move fact record.

Trajectory horizon (2 Nov 2026). The rules run at every ply of the displayed
line, not at two fixed checkpoints. Each fired claim is tracked as a
trajectory keyed by its entity (rule, side, square or file): the ply where it
first fires is its support ply, and it is kept only if it still holds at the
line's endpoint. A claim that fires at or before the line's immediate horizon
(the first settled ply after the move, 1-3) is "immediate"; a later one is
"envisioned" and names the line move that caused it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import chess

from app.core.commentary.features.envisioned import (
    diff_of_diffs,
    diff_vectors,
    envisioned_for_best_move,
    envisioned_for_played_move,
    is_quiescent,
    probe_trim_enabled,
    trim_envisioned_line_by_probe,
)
from app.core.commentary.features.guid_features import compute_feature_vector_fen
from app.core.commentary.features.guid_features.evidence import (
    evidence_for,
    feature_squares,
    evidence_for,
    evidence_from_features,
)
from app.core.commentary.features.experiment_selectors import (
    horizons_match,
    select_adaptive_horizon,
    select_contrastive_claims,
)
from app.core.commentary import pipeline_flags
from app.core.commentary.features.see import best_capture_gain
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.comment_facts import (
    BestAlternative,
    Claim,
    CommentFacts,
    EnvisionedLine,
    FeatureAdvantage,
    FeatureDiff,
)

from .catalog import order_claims_for_mover, run_rules
from .constants import (
    _MATE_SCORE,
    DECISIVE_CLAIM_CP,
    EVAL_CONCESSION_CP,
    MATERIAL_LOSS_CP,
    MAX_ALTERNATIVE_MERITS,
    MAX_CLAIMS_PER_MOVE,
    MAX_FEATURE_ADVANTAGES,
    MERIT_SUPPRESS_CP,
    THRESHOLDS,
)
from .material import _imbalance
from .realization import _line_feature_series
from .tactical_evidence import CONCRETE_IDS, replay, tactical_claims
from .verdicts import verdict_for_eval, verdict_for_transition

logger = logging.getLogger(__name__)

# Cheap depth-limited engine eval of a FEN -> White-POV cp (or None when the
# probe is unavailable/failed). Same signature as
# ``envisioned.trim_envisioned_line_by_probe`` expects.
Prober = Callable[[str], "int | None"]

MATERIAL_IDS = {"material_won", "material_standing", "tactical_material_loss"}
# Static board facts that must be borne out by the displayed line (or a probe)
# before they are stated: a one-square exchange test is not a tactic.
STATIC_GATED = {"piece_left_en_prise", "hanging_piece", "exchange_loses_material", "piece_trapped", "fork"}
# Activity-style claims that a material-changing capture at the move crowds out.
ACTIVITY_IDS = {"activity_improved", "activity_reduced", "rooks_activated", "king_under_pressure",
                "king_safer", "piece_pinned"}
# When several rules describe the same board entity, the first listed wins.
ENTITY_PRIORITY = [
    ("hole", ["outpost_occupied", "strong_knight_established", "hole_created"]),
    ("outpost", ["outpost_occupied", "strong_knight_established"]),
    ("en_prise", ["piece_left_en_prise", "hanging_piece", "target"]),
    ("hanging", ["piece_left_en_prise", "hanging_piece", "intent_targets_piece"]),
    ("target", ["piece_left_en_prise", "hanging_piece", "intent_targets_piece"]),
    ("king_attack", ["king_attack_builds", "king_zone_pressure", "king_under_pressure",
                     "king_shelter_weakened", "king_file_opened"]),
    ("pin", ["absolute_pin", "piece_pinned"]),
]
PROBE_TOLERANCE_CP = 40
_SQUARE = re.compile(r"\b([a-h][1-8])\b")


def _decode_eval(cp: int | None) -> tuple:
    """Split a mate-encoded engine score into (cp, mate)."""
    if cp is None:
        return None, None
    if abs(cp) > _MATE_SCORE - 1000:
        n = _MATE_SCORE - abs(cp)
        return None, (n if cp > 0 else -n)
    return int(cp), None


@dataclass
class _FactsCtx:
    """Shared per-move inputs for the CommentFacts builder helpers."""

    move_event: MoveEvent
    row: AnalyzedMoveData
    board_before: chess.Board
    mover: str
    phase_raw: str
    depth: int
    start_vec: dict
    # Optional cheap engine prober for the probe trim and for confirming
    # static exchange facts. None disables both (engine-free callers/tests).
    prober: Prober | None = None
    horizon_policy: str = "current_quiet_bounded"
    claim_policy: str = "current"
    evidence_suppressions: list[dict] = field(default_factory=list)


def _trim_line(line: EnvisionedLine, prober: Prober | None) -> EnvisionedLine:
    """Apply the probe-based tail trim right after an envisioned line is
    built, BEFORE it backs any feature vector/diff/claim computation."""
    if prober is None or not probe_trim_enabled():
        return line
    try:
        return trim_envisioned_line_by_probe(line, prober)
    except Exception:
        logger.debug("probe trim failed for leaf %s", line.leaf_fen, exc_info=True)
        return line


def _first_quiet_fen(line: EnvisionedLine) -> str:
    """The position at the line's immediate horizon (the first settled ply after
    the move); the leaf when the line never settles within three plies."""
    k = line.immediate_horizon_ply
    if 1 <= k <= len(line.fens):
        return line.fens[k - 1]
    return line.leaf_fen


def claim_identity(claim: Claim) -> tuple:
    """Stable identity of what a claim is about: rule, side and entity."""
    return (claim.rule_id, claim.beneficiary, claim.entity or claim.text)


def _entity_square(claim: Claim) -> str | None:
    if claim.entity and ":" in claim.entity:
        tail = claim.entity.rsplit(":", 1)[1]
        if _SQUARE.fullmatch(tail):
            return tail
    return None


def _dedupe_entities(claims: list[Claim]) -> list[Claim]:
    """Keep one claim per board entity when several rules describe the same square."""
    kept: list[Claim] = []
    rank = {rule: i for _, rules in ENTITY_PRIORITY for i, rule in enumerate(rules)}
    groups = {rule: tuple(rules) for _, rules in ENTITY_PRIORITY for rule in rules}
    by_square: dict[tuple[tuple, str], Claim] = {}
    for claim in sorted(claims, key=lambda c: rank.get(c.rule_id, 99)):
        square = _entity_square(claim)
        group = groups.get(claim.rule_id)
        if square is None or group is None:
            kept.append(claim)
            continue
        key = (group, square)
        if key in by_square:
            continue
        by_square[key] = claim
        kept.append(claim)
    return kept


def _tag(claim: Claim, line: EnvisionedLine, first: int, last: int, immediate_ply: int) -> Claim:
    claim.support_ply = first
    claim.support_fen = line.fens[first - 1]
    claim.baseline_fen = line.start_fen
    claim.first_ply, claim.last_ply = first, last
    claim.persistence = "held" if last == len(line.fens) else "transient"
    # What happens on the move itself (ply 1) is always immediate, even when the
    # sequence never settles; otherwise the immediate horizon decides.
    if first <= max(1, immediate_ply):
        claim.realization = "immediate"
        claim.caused_by_ply, claim.caused_by_san = None, None
    else:
        claim.realization = "envisioned"
        claim.caused_by_ply, claim.caused_by_san = first, line.line_san[first - 1]
    return claim


def _capture_squares(line: EnvisionedLine, boards: list[chess.Board], upto: int) -> list[int]:
    out = []
    for k in range(1, min(upto, len(line.line_uci)) + 1):
        move = chess.Move.from_uci(line.line_uci[k - 1])
        if boards[k - 1].is_capture(move):
            out.append(move.to_square)
    return out


def _fill_evidence(claim: Claim, line: EnvisionedLine, boards: list[chess.Board]) -> None:
    """Derive evidence from the claimed features when the rule gave none."""
    if claim.evidence.squares or claim.evidence.files:
        return
    support = claim.support_ply or len(boards) - 1
    node = boards[min(support, len(boards) - 1)]
    if claim.rule_id in MATERIAL_IDS:
        squares = _capture_squares(line, boards, support)
        claim.evidence = evidence_for(node, squares, baseline_squares=squares, source="rule")
        return
    claim.evidence = evidence_from_features(node, boards[0], claim.features_involved)


def _target_square(claim: Claim) -> int | None:
    for piece in claim.evidence.pieces:
        if piece.get("role") == "target":
            return chess.parse_square(piece["square"])
    if claim.evidence.squares:
        return chess.parse_square(claim.evidence.squares[0])
    return None


def _captured_later(line: EnvisionedLine, boards: list[chess.Board], square: int, after_ply: int) -> bool:
    for k in range(after_ply + 1, len(line.line_uci) + 1):
        move = chess.Move.from_uci(line.line_uci[k - 1])
        if move.to_square == square and boards[k - 1].is_capture(move):
            return True
        if move.from_square == square:
            return False  # the piece left; later captures on the square are of something else
    return False


def _probe_confirms_capture(ctx: _FactsCtx, line: EnvisionedLine, board: chess.Board, square: int,
                            attacker: chess.Color) -> bool | None:
    """Ask the engine whether taking on ``square`` keeps the attacker's eval (None: unknown)."""
    if ctx.prober is None or line.root_eval_cp is None or abs(line.root_eval_cp) >= 999000:
        return None
    if board.turn != attacker:
        return None
    gain, capture = best_capture_gain(board, square)
    if capture is None:
        return False
    probe = board.copy(stack=False)
    probe.push(capture)
    value = ctx.prober(probe.fen())
    if value is None:
        return None
    sign = 1 if attacker == chess.WHITE else -1
    return sign * value >= sign * line.root_eval_cp - PROBE_TOLERANCE_CP


def _gate_static_facts(ctx: _FactsCtx, claims: list[Claim], line: EnvisionedLine,
                       boards: list[chess.Board]) -> list[Claim]:
    """Keep a static exchange fact only when the displayed line (or a probe) bears it out."""
    kept = []
    for claim in claims:
        if claim.rule_id not in STATIC_GATED:
            kept.append(claim)
            continue
        support = claim.support_ply or 1
        square = _target_square(claim)
        reason = None
        if claim.rule_id in ("piece_left_en_prise", "hanging_piece", "piece_trapped"):
            attacker = chess.WHITE if claim.beneficiary == "white" else chess.BLACK
            if square is None:
                reason = "no_target_square"
            elif not _captured_later(line, boards, square, support):
                confirmed = _probe_confirms_capture(ctx, line, boards[support], square, attacker)
                if confirmed is not True:
                    reason = "capture_not_realized_in_line" if confirmed is None else "capture_refuted_by_probe"
        elif claim.rule_id == "exchange_loses_material":
            sign = 1 if ctx.mover == "White" else -1
            series = _line_feature_series(line.start_fen, line.fens, names=["MATERIAL_BALANCE"]).get("MATERIAL_BALANCE") or []
            lost = any(sign * (v - series[0]) <= -MATERIAL_LOSS_CP for v in series[1:4]) if series else False
            if square is None or not (_captured_later(line, boards, square, support) or lost):
                reason = "loss_not_realized_in_line"
        elif claim.rule_id == "fork":
            forker = chess.parse_square(claim.evidence.squares[0]) if claim.evidence.squares else None
            subject = next((chess.parse_square(p["square"]) for p in claim.evidence.pieces if p.get("role") == "subject"), forker)
            targets = [chess.parse_square(p["square"]) for p in claim.evidence.pieces if p.get("role") == "target"]
            if subject is not None and support < len(line.line_uci):
                reply = chess.Move.from_uci(line.line_uci[support])
                if reply.to_square == subject and boards[support].is_capture(reply):
                    reason = "forking_piece_captured_at_once"
            if reason is None:
                king_target = any(boards[support].piece_type_at(t) == chess.KING for t in targets)
                cashed = any(_captured_later(line, boards, t, support) for t in targets)
                # A fork that fails is still the idea behind an error, so the
                # comment can say why it does not work; a sound move's fork
                # must show its gain in the line.
                quality = getattr(ctx.move_event.move_quality, "value", ctx.move_event.move_quality) or ""
                failed_attempt = (line.line_uci[:1] == [ctx.move_event.uci]
                                  and quality in ("inaccuracy", "mistake", "blunder"))
                if not (king_target or cashed or failed_attempt):
                    if pipeline_flags.enabled("fork_probe"):
                        # Like the other static facts: after the reply, taking one of the
                        # targets must keep the forking side's eval within tolerance.
                        attacker = chess.WHITE if claim.beneficiary == "white" else chess.BLACK
                        confirmed = None
                        if support < len(boards) - 1:
                            results = [_probe_confirms_capture(ctx, line, boards[support + 1], t, attacker)
                                       for t in targets]
                            confirmed = True if True in results else (None if None in results else False)
                        if confirmed is not True:
                            reason = ("fork_not_cashed_in_line" if confirmed is None
                                      else "fork_refuted_by_probe")
                    else:
                        reason = "fork_not_cashed_in_line"
        if reason:
            ctx.evidence_suppressions.append({"rule_id": claim.rule_id, "reason": reason,
                                              "support_ply": support, "branch": line.line_uci[:1],
                                              "text": claim.text})
        else:
            kept.append(claim)
    return kept


def _material_changing_capture(claims: list[Claim], boards: list[chess.Board]) -> bool:
    """A capture at the move or the reply that changes the material balance."""
    if any(c.rule_id in ("exchange_loses_material", "exchange_sequence") and (c.support_ply or 9) <= 2 for c in claims):
        return True
    for c in claims:
        if c.rule_id == "legal_capture" and (c.support_ply or 9) <= 2 and len(boards) > c.support_ply:
            if _imbalance(boards[c.support_ply]) != _imbalance(boards[0]):
                return True
    return False


def _trajectory_claims(
    ctx: _FactsCtx, line: EnvisionedLine, eval_cp: int | None
) -> tuple[list[Claim], dict]:
    """Run the rules along the whole displayed line and keep persistent claims.

    Returns the merged claims (concrete facts first, uncapped) and the leaf
    vector (for the material-loss fallback)."""
    leaf_vec = compute_feature_vector_fen(line.leaf_fen)
    boards = replay(line)
    if not boards:
        return [], leaf_vec
    immediate_ply = line.immediate_horizon_ply
    leaf_board = chess.Board(line.leaf_fen)
    concrete = tactical_claims(line)
    for claim in concrete:
        _tag(claim, line, claim.support_ply or 1, len(line.fens), immediate_ply)
        claim.persistence = "held"
    if (
        not line.evidence_valid
        or immediate_ply < 1
        or not is_quiescent(ctx.board_before)
        or not is_quiescent(leaf_board)
        or leaf_board.is_game_over()
    ):
        return _gate_static_facts(ctx, concrete, line, boards), leaf_vec
    common = {
        "phase": ctx.phase_raw,
        "mover": ctx.mover.upper(),
        "eval_cp": eval_cp,
        "start_board": ctx.board_before,
        "cap": False,
    }
    n = len(line.fens)
    trajectories: dict[tuple, dict] = {}
    for k in range(immediate_ply, n + 1):
        node_vec = leaf_vec if k == n else compute_feature_vector_fen(line.fens[k - 1])
        for claim in run_rules(diff_vectors(ctx.start_vec, node_vec), leaf_board=boards[k], **common):
            key = claim_identity(claim)
            entry = trajectories.setdefault(key, {"first": k, "claim": claim})
            entry["last"] = k
    kept: list[Claim] = []
    quality = getattr(ctx.move_event.move_quality, "value", ctx.move_event.move_quality) or ""
    # At an error, a transient claim in the opponent's favour that lasts at least
    # two plies is the point of the error (Guid, p. 76-77): the opponent gains
    # something that is later traded off, and the comparison with the line's end
    # would hide it. Kept with persistence "transient" when the switch is on.
    keep_transient = (pipeline_flags.enabled("transient_at_errors")
                      and line.line_uci[:1] == [ctx.move_event.uci]
                      and quality in ("inaccuracy", "mistake", "blunder"))
    for entry in trajectories.values():
        claim = _tag(entry["claim"], line, entry["first"], entry["last"], immediate_ply)
        if (claim.persistence != "held" and keep_transient
                and claim.beneficiary not in (ctx.mover.lower(), None)
                and entry["last"] - entry["first"] + 1 >= 2):
            _fill_evidence(claim, line, boards)
            kept.append(claim)
            continue
        if claim.persistence != "held":
            ctx.evidence_suppressions.append({
                "rule_id": claim.rule_id, "reason": "transient_in_displayed_line",
                "support_ply": entry["first"], "last_ply": entry["last"],
                "branch": line.line_uci[:1], "text": claim.text})
            continue
        _fill_evidence(claim, line, boards)
        kept.append(claim)
    kept = _dedupe_entities(kept)
    kept = _gate_static_facts(ctx, kept, line, boards)
    concrete = _gate_static_facts(ctx, concrete, line, boards)
    if _material_changing_capture(concrete, boards):
        # A capture that changes material explains the move; the activity
        # swing it causes is a side effect, not a plan.
        dropped = [c for c in kept if c.rule_id in ACTIVITY_IDS]
        for c in dropped:
            ctx.evidence_suppressions.append({"rule_id": c.rule_id, "reason": "activity_crowded_out_by_capture",
                                              "support_ply": c.support_ply, "branch": line.line_uci[:1], "text": c.text})
        kept = [c for c in kept if c.rule_id not in ACTIVITY_IDS]
    return concrete + kept, leaf_vec


# Backwards-compatible alias (the two-checkpoint builder was replaced by trajectories).
_two_horizon_claims = _trajectory_claims


def _played_line_claims(
    ctx: _FactsCtx,
) -> tuple[EnvisionedLine, FeatureDiff, dict, list[Claim], dict]:
    """Envisioned played line + its feature diff, leaf vector, and ordered claims."""
    move_event = ctx.move_event
    engine_meta = (ctx.row.hidden_features or {}).get("_engine") or {}
    after_pv = list(engine_meta.get("after_pv_uci") or [])
    if ctx.horizon_policy == "adaptive_feature":
        played_line, selection = select_adaptive_horizon(
            move_event.fen_before,
            [move_event.uci, *after_pv],
            root_eval_cp=move_event.eval_after_cp,
            depth=ctx.depth,
            prober=ctx.prober,
        )
    else:
        played_line = envisioned_for_played_move(
            move_event.fen_before,
            move_event.uci,
            after_pv,
            played_eval_cp=move_event.eval_after_cp,
            depth=ctx.depth,
        )
        played_line = _trim_line(played_line, ctx.prober)
        selection = {"version": "trajectory_settlement_v1", "selected_ply": len(played_line.fens),
                     "reason": played_line.horizon_reason, "settled_ply": played_line.settled_ply,
                     "stable_ply": played_line.stable_ply,
                     "immediate_horizon_ply": played_line.immediate_horizon_ply}
    claims, leaf_vec = _trajectory_claims(ctx, played_line, move_event.eval_after_cp)
    line_diff = diff_vectors(ctx.start_vec, leaf_vec)
    ordered = claims if ctx.claim_policy == "contrastive" else order_claims_for_mover(claims, ctx.mover)
    return played_line, line_diff, leaf_vec, _concrete_first(ordered), selection


def _concrete_first(claims: list[Claim]) -> list[Claim]:
    """Event priority is independent of feature magnitudes or beneficiary."""
    return sorted(claims, key=lambda c: (
        0 if c.rule_id in CONCRETE_IDS else 1,
        (c.support_ply or 0) if c.rule_id in CONCRETE_IDS else 0,
    ))


def _net_material_cp(vec: dict) -> int:
    fv = vec.get("MATERIAL_BALANCE")
    return fv.value_cp if fv is not None else 0


def _fallback_claim(
    ctx: _FactsCtx, leaf_vec: dict, refutation_san: str | None
) -> Claim | None:
    """One grounded claim (material loss / engine preference) when no rule fired."""
    move_event = ctx.move_event
    opp = "black" if ctx.mover == "White" else "white"
    mover_sign = 1 if ctx.mover == "White" else -1
    mat_delta = mover_sign * (
        _net_material_cp(leaf_vec) - _net_material_cp(ctx.start_vec)
    )
    best_san = move_event.best_move_san
    if mat_delta <= -MATERIAL_LOSS_CP:
        text = f"{ctx.mover} loses material"
        if best_san:
            text += f"; {best_san} held the balance"
        return Claim(
            rule_id="tactical_material_loss",
            text=text + ".",
            beneficiary=opp,
            delta_cp=abs(mat_delta),
            features_involved=["MATERIAL_BALANCE"],
        )
    if refutation_san is None and best_san:
        return Claim(
            rule_id="eval_concession",
            text=f"the engine preferred {best_san} here.",
            beneficiary=opp,
            delta_cp=EVAL_CONCESSION_CP,
        )
    return None


# Plain names of the feature groups a merit or edge claim may speak of.
FEATURE_GROUPS = (("MATERIAL", "material"), ("KING_DANGER", "king safety"), ("KING_SAFETY", "king safety"),
                  ("PASSED", "passed pawns"), ("BISHOP_PAIR", "the bishop pair"), ("BISHOP", "bishop quality"),
                  ("ROOK", "rook activity"), ("KNIGHT", "knight placement"), ("PAWN", "pawn structure"),
                  ("MOBILITY", "piece activity"), ("ACTIVITY", "piece activity"), ("SPACE", "space"),
                  ("CENTER", "central control"), ("KING", "king safety"))
DIVERGENCE_MIN_CP = 30  # a feature group difference that already distinguishes the lines
MERIT_MIN_CP = 10  # smallest feature edge worth stating as the point of a sound move
EDGE_MIN_CP = 4  # smallest matched-depth difference stated as what the alternative does better


def feature_group(name: str) -> str | None:
    return next((label for key, label in FEATURE_GROUPS if key in name), None)


def _feature_claim(rule_id: str, text: str, beneficiary: str, name: str, cp: int,
                   line: EnvisionedLine, ply: int) -> Claim:
    """A claim about one feature at ``ply`` of ``line``, with evidence from the feature's definition."""
    ply = max(1, min(ply, len(line.fens)))
    board = chess.Board(line.fens[ply - 1])
    squares, files = feature_squares(board, name)
    claim = Claim(rule_id=rule_id, text=text, text_state=text, beneficiary=beneficiary, supplementary=True,
                  features_involved=[name], delta_cp=abs(cp), entity=f"{rule_id}:{feature_group(name)}",
                  evidence=evidence_for(board, squares, files=files, source="feature"))
    return _tag(claim, line, ply, len(line.fens), line.immediate_horizon_ply)


def merit_claim(ctx: _FactsCtx, claims: list[Claim], line: EnvisionedLine, diff: FeatureDiff,
                better: BestAlternative | None) -> Claim | None:
    """The point of a sound move when no rule produced a merit for the mover.

    First choice: the feature group where the played line beats the alternative
    (matched depth when available). Otherwise the mover's largest feature gain in
    the played line itself. None when neither reaches the threshold."""
    mover_key = ctx.mover.lower()
    if any(c.beneficiary == mover_key and not c.is_concession for c in claims):
        return None
    san = ctx.move_event.san
    sign = 1 if ctx.mover == "White" else -1
    if better is not None:
        matched = bool(better.matched_ply and better.feature_advantages_matched)
        advantages = better.feature_advantages_matched if matched else better.feature_advantages
        favouring = [a for a in advantages
                     if sign * a.advantage_cp < 0 and feature_group(a.name) and "MATERIAL" not in a.name]
        if favouring:
            best = max(favouring, key=lambda a: abs(a.advantage_cp))
            if abs(best.advantage_cp) >= MERIT_MIN_CP:
                text = (f"Compared with {better.san}, {san} leaves {ctx.mover} better placed in "
                        f"{feature_group(best.name)}.")
                return _feature_claim("merit_over_alternative", text, mover_key, best.name, best.advantage_cp,
                                      line, better.matched_ply if matched else len(line.fens))
    gains = [d for d in (diff.positive if ctx.mover == "White" else diff.negative)
             if feature_group(d.name) and "MATERIAL" not in d.name]
    if gains:
        best_delta = max(gains, key=lambda d: abs(d.delta_cp))
        if abs(best_delta.delta_cp) >= MERIT_MIN_CP:
            text = f"The main gain of {san} for {ctx.mover} is {feature_group(best_delta.name)}."
            return _feature_claim("main_gain", text, mover_key, best_delta.name, best_delta.delta_cp,
                                  line, len(line.fens))
    return None


def alternative_edge_claim(ctx: _FactsCtx, alt_san: str, alt_line: EnvisionedLine,
                           matched_ply: int | None, played_line: EnvisionedLine) -> Claim | None:
    """What the better move does better when the regular comparison is empty: the
    largest matched-depth feature difference in the alternative's favour, even a small one."""
    if matched_ply is None or matched_ply < 1:
        return None
    played_vec = compute_feature_vector_fen(played_line.fens[matched_ply - 1])
    alt_vec = compute_feature_vector_fen(alt_line.fens[matched_ply - 1])
    advantages = diff_of_diffs(diff_vectors(ctx.start_vec, played_vec), diff_vectors(ctx.start_vec, alt_vec),
                               min_abs_cp=EDGE_MIN_CP, max_features=12)
    sign = 1 if ctx.mover == "White" else -1
    favouring = [a for a in advantages if sign * a.advantage_cp > 0 and feature_group(a.name)]
    if not favouring:
        return None
    best = max(favouring, key=lambda a: abs(a.advantage_cp))
    text = (f"Where {alt_san} does better than {ctx.move_event.san}: {feature_group(best.name)}, "
            f"{matched_ply} plies after the move.")
    return _feature_claim("alternative_edge", text, ctx.mover.lower(), best.name, best.advantage_cp,
                          alt_line, matched_ply)


def lines_diverge_claim(ctx: _FactsCtx, alt_san: str, alt_line: EnvisionedLine,
                        played_line: EnvisionedLine) -> Claim | None:
    """When no feature separates the two moves: the first point where the lines part, i.e. the
    first reply or follow-up that is different, named for both moves. Grounded in the traces only."""
    played, alt = played_line.line_san, alt_line.line_san
    if len(alt) < 2:
        return None
    i = 1
    while i < len(played) and i < len(alt) and played[i] == alt[i]:
        i += 1
    if i >= len(alt):
        return None
    side = ctx.mover if i % 2 == 0 else ("Black" if ctx.mover == "White" else "White")
    alt_part = " ".join(alt[i:i + 2])
    played_part = " ".join(played[i:i + 2]) if i < len(played) else None
    if played_part:
        text = (f"The lines part at {side}'s next move: after {alt_san} it continues {alt_part}, "
                f"after {ctx.move_event.san} it is {played_part}.")
    else:
        text = (f"The lines part after the shared reply: {alt_san} allows {side} to continue with "
                f"{alt_part}, which {ctx.move_event.san} does not.")
    ply = min(i + 1, len(alt_line.fens))
    board = chess.Board(alt_line.fens[ply - 1])
    move = chess.Move.from_uci(alt_line.line_uci[ply - 1])
    claim = Claim(rule_id="lines_diverge", text=text, text_state=text, beneficiary=ctx.mover.lower(),
                  delta_cp=EDGE_MIN_CP, entity="lines_diverge", supplementary=True,
                  evidence=evidence_for(board, [move.from_square, move.to_square], roles={move.to_square: "subject"}))
    return _tag(claim, alt_line, ply, len(alt_line.fens), alt_line.immediate_horizon_ply)


def _cap_positional(claims: list[Claim], other: set[tuple], mover: str, limit: int) -> list[Claim]:
    """Contrast-first cap: concrete and material claims are exempt; among the
    rest, claims about entities the other line does not share rank first,
    then by weight. Concessions keep the order ``order_claims_for_mover`` gave."""
    exempt = [c for c in claims if c.rule_id in CONCRETE_IDS or c.rule_id in MATERIAL_IDS]
    rest = [c for c in claims if c not in exempt]
    rest.sort(key=lambda c: (c.is_concession, claim_identity(c) in other, -c.delta_cp))
    kept = exempt + rest[:limit]
    return [c for c in claims if c in kept]


def _alternative_merits(
    best_claims: list[Claim], main_claims: list[Claim], mover: str, claim_policy: str = "current"
) -> list[Claim]:
    """The alternative's own merits (max 2), never repeating the main block's claims."""
    concrete = [c.model_copy(update={"is_concession": c.beneficiary not in (mover.lower(), None)})
                for c in _concrete_first(best_claims) if c.rule_id in CONCRETE_IDS][:3]
    if claim_policy == "contrastive":
        return concrete + [c for c in best_claims if c.rule_id not in CONCRETE_IDS]
    main_ids = {claim_identity(c) for c in main_claims}
    mover_key = mover.lower()
    rest = [c for c in best_claims
            if c.rule_id not in CONCRETE_IDS and c.beneficiary in (mover_key, None) and claim_identity(c) not in main_ids]
    rest.sort(key=lambda c: (c.rule_id not in MATERIAL_IDS, -c.delta_cp))
    return concrete + rest[:max(0, MAX_ALTERNATIVE_MERITS - len(concrete))]


def _feature_advantages(
    played_diff: FeatureDiff | None, alt_diff: FeatureDiff
) -> list[FeatureAdvantage]:
    """Feature-diff-of-diffs: the alternative's own start->leaf swing vs. the
    played move's, per feature, in cp."""
    if played_diff is None:
        return []
    return diff_of_diffs(
        played_diff,
        alt_diff,
        min_abs_cp=THRESHOLDS.get("min_claim_cp", 8),
        max_features=MAX_FEATURE_ADVANTAGES,
    )


def _matched_comparison(ctx: _FactsCtx, played: EnvisionedLine, alt: EnvisionedLine
                        ) -> tuple[int | None, list[FeatureAdvantage], str | None]:
    """Compare both lines at the same depth (the shorter line's length) as well.

    With the ``quiet_end_comparison`` switch the lines are compared only at their
    own quiet ends (``_feature_advantages``), because the longer line is not
    checked for quiescence at the shorter line's length."""
    n = min(len(played.fens), len(alt.fens))
    if n < 1 or pipeline_flags.enabled("quiet_end_comparison"):
        return None, [], None
    played_vec = compute_feature_vector_fen(played.fens[n - 1])
    alt_vec = compute_feature_vector_fen(alt.fens[n - 1])
    matched = diff_of_diffs(diff_vectors(ctx.start_vec, played_vec), diff_vectors(ctx.start_vec, alt_vec),
                            min_abs_cp=THRESHOLDS.get("min_claim_cp", 8), max_features=MAX_FEATURE_ADVANTAGES)
    note = None
    if abs(len(played.fens) - len(alt.fens)) > 1:
        note = (f"The lines settle at different depths: {len(played.fens)} plies after "
                f"{played.line_san[0] if played.line_san else '?'} and {len(alt.fens)} plies after "
                f"{alt.line_san[0] if alt.line_san else '?'}; the comparison at {n} plies is also given.")
    return n, matched, note


def _build_alternative_line(ctx: _FactsCtx, pv_uci: list[str], eval_cp: int | None) -> EnvisionedLine:
    move_event = ctx.move_event
    if ctx.horizon_policy == "adaptive_feature":
        line, _selection = select_adaptive_horizon(
            move_event.fen_before, pv_uci, root_eval_cp=eval_cp, depth=ctx.depth, prober=ctx.prober,
        )
        return line
    line = envisioned_for_best_move(move_event.fen_before, pv_uci, best_eval_cp=eval_cp, depth=ctx.depth)
    return _trim_line(line, ctx.prober)


def _build_better_alternative(
    ctx: _FactsCtx, main_claims: list[Claim], played_diff: FeatureDiff | None = None,
    played_line: EnvisionedLine | None = None,
) -> BestAlternative | None:
    """The engine's preferred move + its merits, whenever a different move was
    played — so the better line can always be visualized (Guid)."""
    move_event = ctx.move_event
    if not (
        move_event.best_move_uci
        and move_event.best_move_uci != move_event.uci
        and move_event.best_move_eval_cp is not None
        and move_event.eval_after_cp is not None
    ):
        return None

    best_pv_uci = (
        [str(m.move) for m in ctx.row.pvs[0] if getattr(m, "move", None)]
        if ctx.row.pvs and ctx.row.pvs[0]
        else []
    )
    best_cp, best_mate = _decode_eval(move_event.best_move_eval_cp)
    best_line = _build_alternative_line(ctx, best_pv_uci, best_cp)
    best_claims, best_leaf_vec = _trajectory_claims(ctx, best_line, best_cp)
    if _is_decided(best_cp, best_mate):
        best_claims = _material_only(best_claims)
    alt_diff = diff_vectors(ctx.start_vec, best_leaf_vec)
    matched_ply, matched, note = _matched_comparison(ctx, played_line, best_line) if played_line else (None, [], None)
    alt_san = move_event.best_move_san or move_event.best_move_uci
    merits = _alternative_merits(best_claims, main_claims, ctx.mover, ctx.claim_policy)
    sign = 1 if ctx.mover == "White" else -1
    distinguishing = [a for a in (matched or _feature_advantages(played_diff, alt_diff))
                      if sign * a.advantage_cp >= DIVERGENCE_MIN_CP]
    main_ids = {claim_identity(c) for c in main_claims}
    fresh = [c for c in merits if c.beneficiary == ctx.mover.lower() and claim_identity(c) not in main_ids]
    if played_line and not distinguishing and not fresh:
        edge = (alternative_edge_claim(ctx, alt_san, best_line, matched_ply, played_line)
                or lines_diverge_claim(ctx, alt_san, best_line, played_line))
        if edge is not None:
            merits = merits + [edge]
    return BestAlternative(
        san=alt_san,
        uci=move_event.best_move_uci,
        eval_cp=best_cp,
        eval_mate=best_mate,
        verdict=verdict_for_eval(best_cp, best_mate),
        display_line=best_line,
        claims=merits,
        feature_advantages=_feature_advantages(played_diff, alt_diff),
        matched_ply=matched_ply,
        feature_advantages_matched=matched,
        horizon_note=note,
    )


def _build_inferior_alternative(
    ctx: _FactsCtx, main_claims: list[Claim], played_diff: FeatureDiff | None = None,
    played_line: EnvisionedLine | None = None,
) -> BestAlternative | None:
    """When the engine's best move was actually played, surface the runner-up
    (2nd-best PV) as a contrast for the "!" (Guid)."""
    move_event = ctx.move_event
    if not (move_event.best_move_uci and move_event.best_move_uci == move_event.uci):
        return None
    if move_event.eval_after_cp is None:
        return None
    pvs = ctx.row.pvs or []
    second = pvs[1][0] if len(pvs) >= 2 and pvs[1] else None
    if second is None or getattr(second, "move", None) is None or second.score is None:
        return None

    played_cp, _ = _decode_eval(move_event.eval_after_cp)
    second_cp, second_mate = _decode_eval(second.score)
    if played_cp is None or second_cp is None:  # mate-coded line: skip the contrast
        return None
    second_uci = str(second.move)
    if second_uci == move_event.uci:  # degenerate: 2nd PV repeats the played move
        return None
    try:
        second_san = ctx.board_before.san(chess.Move.from_uci(second_uci))
    except (ValueError, chess.IllegalMoveError):
        second_san = second_uci
    second_pv_uci = [str(m.move) for m in pvs[1] if getattr(m, "move", None)]
    alt_line = _build_alternative_line(ctx, second_pv_uci, second_cp)
    alt_claims, alt_leaf_vec = _trajectory_claims(ctx, alt_line, second_cp)
    if _is_decided(second_cp, second_mate):
        alt_claims = _material_only(alt_claims)
    alt_diff = diff_vectors(ctx.start_vec, alt_leaf_vec)
    matched_ply, matched, note = _matched_comparison(ctx, played_line, alt_line) if played_line else (None, [], None)
    return BestAlternative(
        san=second_san,
        uci=second_uci,
        eval_cp=second_cp,
        eval_mate=second_mate,
        verdict=verdict_for_eval(second_cp, second_mate),
        display_line=alt_line,
        claims=_alternative_merits(alt_claims, main_claims, ctx.mover, ctx.claim_policy),
        is_inferior=True,
        feature_advantages=_feature_advantages(played_diff, alt_diff),
        matched_ply=matched_ply,
        feature_advantages_matched=matched,
        horizon_note=note,
    )


def _with_feature_series(line: EnvisionedLine, fen_before: str) -> EnvisionedLine:
    """Attach the full per-feature progression along the line (for the charts)."""
    return line.model_copy(
        update={"feature_series": _line_feature_series(fen_before, line.fens)}
    )


def _refutation_san(played_line: EnvisionedLine, move_quality: str) -> str | None:
    """The opponent's punishing reply — only meaningful for a real mistake."""
    if move_quality in ("mistake", "blunder") and len(played_line.line_san) >= 2:
        return played_line.line_san[1]
    return None


_DUBIOUS = ("inaccuracy", "mistake", "blunder")
SOUND_GAP_CP = 50  # a "good" move that is this much behind the engine choice is not given a merit
SOUND_QUALITIES = {"best", "best_move", "good", "great", "great_move", "brilliant", "excellent", "book"}


ONLY_WIN_MARGIN_CP = 10_000  # first choice keeps a forced mate, the second does not


def _best_margin_cp(pvs: list, mover: str) -> int | None:
    """Mover-POV gap between the engine's first and second choice (same search).

    MultiPV lines are ordered, so this bounds the gap to every other move."""
    if len(pvs or []) < 2 or not pvs[0] or not pvs[1]:
        return None
    first, second = pvs[0][0], pvs[1][0]
    if first.score is None or second.score is None:
        return None
    sign = 1 if mover == "White" else -1
    (best_cp, best_mate), (second_cp, second_mate) = _decode_eval(first.score), _decode_eval(second.score)
    if best_mate is not None or second_mate is not None:
        best_wins = best_mate is not None and best_mate * sign > 0
        second_wins = second_mate is not None and second_mate * sign > 0
        return ONLY_WIN_MARGIN_CP if best_wins and not second_wins else None
    return sign * (best_cp - second_cp)


def _concession_mode(move_quality: str) -> str:
    """For dubious moves opponent-favoring claims explain the error, not trade-offs."""
    return "consequence" if move_quality in _DUBIOUS else "tradeoff"


def _is_decided(eval_cp: int | None, eval_mate: int | None) -> bool:
    return eval_mate is not None or (
        eval_cp is not None and abs(eval_cp) > DECISIVE_CLAIM_CP
    )


def _material_only(claims: list[Claim]) -> list[Claim]:
    """In a decided position positional claims are noise (a passed pawn in
    mate-in-4); only the material standing still explains the result."""
    return [c for c in claims if "MATERIAL_BALANCE" in c.features_involved or c.rule_id in CONCRETE_IDS]


def _filter_claims(
    claims: list[Claim],
    *,
    eval_cp: int | None,
    eval_mate: int | None,
    mover: str,
    move_quality: str,
    ctx: _FactsCtx,
    leaf_vec: dict,
    refutation_san: str | None,
) -> list[Claim]:
    """Drop misleading claims: noise in decided positions, and the mover's
    incidental "merits" when the move leaves them clearly worse off."""
    if _is_decided(eval_cp, eval_mate):
        before = claims
        claims = _material_only(claims)
        if len(claims) < len(before):
            logger.debug(
                "_filter_claims: decisive-position gate (eval_cp=%s, eval_mate=%s, "
                "threshold=%d) dropped %d non-material claim(s): %s",
                eval_cp, eval_mate, DECISIVE_CLAIM_CP, len(before) - len(claims),
                [c.rule_id for c in before if c not in claims],
            )
    mover_pov_eval = (
        None if eval_cp is None else (eval_cp if mover == "White" else -eval_cp)
    )
    if mover_pov_eval is not None and mover_pov_eval <= -MERIT_SUPPRESS_CP:
        mover_key = mover.lower()
        before = claims
        claims = [
            c
            for c in claims
            if c.rule_id in CONCRETE_IDS or c.is_concession or c.beneficiary not in (mover_key, None)
        ]
        if len(claims) < len(before):
            logger.debug(
                "_filter_claims: merit-suppress gate (mover_pov_eval=%s, "
                "threshold=-%d) dropped %d mover-merit claim(s): %s",
                mover_pov_eval, MERIT_SUPPRESS_CP, len(before) - len(claims),
                [c.rule_id for c in before if c not in claims],
            )
    return claims


def _attach_series(
    played_line: EnvisionedLine,
    better: BestAlternative | None,
    fen_before: str,
) -> tuple[EnvisionedLine, BestAlternative | None]:
    """Attach the per-ply feature series to the charted lines."""
    played_line = _with_feature_series(played_line, fen_before)
    if better is not None and better.display_line is not None:
        better = better.model_copy(
            update={
                "display_line": _with_feature_series(better.display_line, fen_before)
            }
        )
    return played_line, better


def build_comment_facts(
    row: AnalyzedMoveData,
    move_event: MoveEvent,
    *,
    depth: int = 16,
    prober: Prober | None = None,
    horizon_policy: str = "current_quiet_bounded",
    claim_policy: str = "current",
) -> CommentFacts | None:
    """Assemble the move's inviolable facts from data the engine pass already paid for.

    ``prober`` is an optional cheap engine callable (FEN -> White-POV cp) used
    to sanity-check envisioned lines' leaves and to confirm static exchange
    facts the displayed line does not realize; omit it for engine-free callers."""
    if horizon_policy not in {"current_quiet_bounded", "adaptive_feature"}:
        raise ValueError(f"unknown horizon policy: {horizon_policy}")
    if claim_policy not in {"current", "contrastive"}:
        raise ValueError(f"unknown claim policy: {claim_policy}")
    phase_raw = row.phase_raw or "mid"
    if phase_raw == "early":
        return None

    board_before = chess.Board(move_event.fen_before)
    board_after = chess.Board(move_event.fen_after)
    outcome = board_after.outcome()
    terminal_result = outcome.result() if outcome is not None else None
    mover = "White" if board_before.turn == chess.WHITE else "Black"
    eval_cp, eval_mate = _decode_eval(move_event.eval_after_cp)
    eval_before_cp, _before_mate = _decode_eval(move_event.eval_before_cp)
    ctx = _FactsCtx(
        move_event=move_event,
        row=row,
        board_before=board_before,
        mover=mover,
        phase_raw=phase_raw,
        depth=depth,
        start_vec=compute_feature_vector_fen(move_event.fen_before),
        prober=prober,
        horizon_policy=horizon_policy,
        claim_policy=claim_policy,
    )

    played_line, diff, leaf_vec, claims, horizon_selection = _played_line_claims(ctx)
    move_quality = move_event.move_quality.value if move_event.move_quality else ""
    refutation_san = _refutation_san(played_line, move_quality)
    claims = _filter_claims(
        claims,
        eval_cp=eval_cp,
        eval_mate=eval_mate,
        mover=mover,
        move_quality=move_quality,
        ctx=ctx,
        leaf_vec=leaf_vec,
        refutation_san=refutation_san,
    )
    better = None if terminal_result else (_build_better_alternative(
        ctx, claims, diff, played_line
    ) or _build_inferior_alternative(ctx, claims, diff, played_line))
    contrast_selection = None
    if better is not None:
        better = better.model_copy(
            update={"horizon_matched": horizons_match(played_line, better.display_line)}
        )
    mover_pov = None if eval_cp is None else (eval_cp if mover == "White" else -eval_cp)
    gap = (None if better is None or better.eval_cp is None or eval_cp is None
           else (1 if mover == "White" else -1) * (better.eval_cp - eval_cp))
    sound = (move_quality in SOUND_QUALITIES and not terminal_result and not _is_decided(eval_cp, eval_mate)
             and mover_pov is not None and mover_pov > -MERIT_SUPPRESS_CP
             and (better is None or better.is_inferior or (gap is not None and gap < SOUND_GAP_CP)))
    if sound:
        merit = merit_claim(ctx, claims, played_line, diff, better)
        if merit is not None:
            claims = claims + [merit]
    if claim_policy == "contrastive" and better is not None:
        played_concrete = [c for c in claims if c.rule_id in CONCRETE_IDS]
        alternative_concrete = [c for c in better.claims if c.rule_id in CONCRETE_IDS]
        claims, alt_claims, contrast_selection = select_contrastive_claims(
            claims, better.claims, mover=mover
        )
        claims = played_concrete + [c for c in claims if c.rule_id not in CONCRETE_IDS]
        alt_claims = alternative_concrete[:3] + [
            c for c in alt_claims if c.rule_id not in CONCRETE_IDS
        ][:max(0, MAX_ALTERNATIVE_MERITS - len(alternative_concrete))]
        better = better.model_copy(update={"claims": _concrete_first(alt_claims)})
    elif claim_policy == "contrastive":
        played_concrete = [c for c in claims if c.rule_id in CONCRETE_IDS]
        claims, _unused, contrast_selection = select_contrastive_claims(claims, [], mover=mover)
        claims = played_concrete + [c for c in claims if c.rule_id not in CONCRETE_IDS]
    else:
        # Contrast-first cap over the uncapped pool: what distinguishes the
        # played line from the alternative ranks above what both share.
        other = {claim_identity(c) for c in better.claims} if better is not None else set()
        claims = _cap_positional(claims, other, mover, MAX_CLAIMS_PER_MOVE)
    claims = _concrete_first(claims)
    played_line, better = _attach_series(played_line, better, move_event.fen_before)
    verdict = verdict_for_transition(eval_before_cp, eval_cp, eval_mate, mover, move_class=move_quality)
    if terminal_result:
        claims = []
        eval_mate = None  # integer zero cannot represent the winning side
        eval_cp = None
        verdict = (f"delivers checkmate for {mover}" if board_after.is_checkmate()
                   else "ends the game in a draw")

    facts = CommentFacts(
        ply=move_event.ply,
        san=move_event.san,
        uci=move_event.uci,
        mover=mover,
        phase=phase_raw,
        verdict=verdict,
        terminal_result=terminal_result,
        move_number=board_before.fullmove_number,
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
        concession_mode=_concession_mode(move_quality),
        move_quality=move_quality or None,
        best_margin_cp=_best_margin_cp(ctx.row.pvs, mover),
        selection_metadata={
            "horizon_policy": horizon_policy,
            "claim_policy": claim_policy,
            "horizon": horizon_selection,
            "contrast": contrast_selection,
            "evidence_suppressions": ctx.evidence_suppressions,
            "alternative_horizon_matched": (
                better.horizon_matched if better is not None else None
            ),
            "alternative_horizon_note": better.horizon_note if better is not None else None,
        },
    )
    if better is not None and pipeline_flags.enabled("scaled_class_limits"):
        # Recorded with the facts, so every reader of the material (template, writer,
        # interface) names the alternative the same way.
        from app.core.commentary.phases.composer.framing import alternative_comparison
        better.comparison = alternative_comparison(facts)
    return facts
