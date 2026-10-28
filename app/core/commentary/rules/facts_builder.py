"""CommentFacts assembly: the played line, its claims, the better alternative,
realization tagging, and the inviolable per-move fact record."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import chess

from app.core.commentary.features.envisioned import (
    diff_of_diffs,
    diff_vectors,
    envisioned_for_best_move,
    envisioned_for_played_move,
    is_quiescent,
    max_display_plies,
    probe_trim_enabled,
    trim_envisioned_line_by_probe,
)
from app.core.commentary.features.guid_features import compute_feature_vector_fen
from app.core.commentary.features.experiment_selectors import (
    horizons_match,
    select_adaptive_horizon,
    select_contrastive_claims,
)
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.comment_facts import (
    BestAlternative,
    Claim,
    CommentFacts,
    EnvisionedLine,
    FeatureAdvantage,
    FeatureDelta,
    FeatureDiff,
)

from .catalog import order_claims_for_mover, run_rules
from .constants import (
    _MATE_SCORE,
    DECISIVE_CLAIM_CP,
    EVAL_CONCESSION_CP,
    MATERIAL_LOSS_CP,
    MAX_ALTERNATIVE_MERITS,
    MAX_FEATURE_ADVANTAGES,
    MERIT_SUPPRESS_CP,
    THRESHOLDS,
)
from .realization import _line_feature_series
from .verdicts import verdict_for_eval, verdict_for_transition
from .tactical_evidence import CONCRETE_IDS, replay, tactical_claims

logger = logging.getLogger(__name__)

# Cheap depth-limited engine eval of a FEN -> White-POV cp (or None when the
# probe is unavailable/failed). Same signature as
# ``envisioned.trim_envisioned_line_by_probe`` expects.
Prober = Callable[[str], "int | None"]


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
    # Optional cheap engine prober for the post-build horizon-effect trim
    # (Improvement 1). None disables the trim (engine-free callers/tests).
    prober: Prober | None = None
    horizon_policy: str = "current_quiet_bounded"
    claim_policy: str = "current"
    evidence_suppressions: list[dict] = field(default_factory=list)


def _trim_line(line: EnvisionedLine, prober: Prober | None) -> EnvisionedLine:
    """Apply the probe-based tail trim right after an envisioned line is
    built, BEFORE it backs any feature vector/diff/claim computation — a
    statically-quiescent leaf can still be tactically loaded (discovered
    attack, skewer), and once claims are computed from it there is no way to
    walk that back. ``ENVISIONED_PROBE_TRIM_ENABLED=0`` opts out; a missing
    prober (engine-free path, most unit tests) is a no-op either way."""
    if prober is None or not probe_trim_enabled():
        return line
    try:
        return trim_envisioned_line_by_probe(line, prober)
    except Exception:
        logger.debug("probe trim failed for leaf %s", line.leaf_fen, exc_info=True)
        return line


def _first_quiet_fen(line: EnvisionedLine) -> str:
    """The first quiescent position at/after the played move — the horizon at
    which the move is *assessed*. For a quiet move that is the position right
    after it; for a capture that will be recaptured it is the node where the
    forcing sequence settles (so transient features are not reported as real)."""
    for fen in line.fens:
        if fen and is_quiescent(chess.Board(fen)):
            return fen
    return line.leaf_fen


def _feature_extremum_ply(series_for_feature: list[int], direction: int, window: int) -> int:
    """Index into ``series_for_feature`` (0 = start position, matching
    ``realization._line_feature_series``'s point-0-is-start convention) of
    the most extreme value in ``direction`` (+1/-1, the sign of the delta
    already fired for this feature) within the first ``window`` plies.

    Guards against the fixed first-quiescent-ply checkpoint landing PAST the
    moment a transient feature (a piece hanging mid-sequence, a rook's open
    file before it gets blocked again) was actually at its most pronounced,
    just because the position stayed non-quiescent a few plies longer for
    unrelated reasons."""
    if not series_for_feature:
        return 0
    limit = min(window, len(series_for_feature) - 1)
    best_idx = 0
    best_val = direction * series_for_feature[0]
    for i in range(1, limit + 1):
        val = direction * series_for_feature[i]
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


def _sharpen_immediate_diff(
    diff: FeatureDiff, line: EnvisionedLine, immediate_fen: str
) -> FeatureDiff:
    """Per-feature horizon fix (Improvement 2): for every feature already in
    the immediate diff, check its full per-ply series (engine-free,
    ``realization._line_feature_series``) for a materially stronger value —
    same direction as the fired delta — earlier in the line than the fixed
    first-quiescent-ply checkpoint, within the line's normal display window.
    When one exists, use THAT ply's value for the claim instead, so the
    claim reflects the moment the feature was actually most true rather than
    an arbitrary quiescence checkpoint that may have overshot it.

    Design choice: we REPLACE the immediate value (rather than only adding a
    flag) because the fired ``delta_cp`` is a concrete number printed in the
    claim text; a flag-only approach would leave that number stale. We only
    ever move further in the ALREADY-fired direction (never introduce a new
    feature, never flip a delta's sign), so a delta can never jump between
    the positive/negative lists here, and the immediate-vs-forecast dedup in
    ``_two_horizon_claims`` (``_is_new``) is keyed on feature name/beneficiary
    — never on the numeric value — so this cannot regress that priority
    ordering; it only changes which FEN backs the number."""
    names = [d.name for d in (diff.positive + diff.negative)]
    if not names:
        return diff
    series = _line_feature_series(line.start_fen, line.fens, names=names)
    points = [line.start_fen, *line.fens]
    try:
        immediate_idx = points.index(immediate_fen)
    except ValueError:
        immediate_idx = len(points) - 1
    window = min(max_display_plies(), len(line.fens))
    threshold = THRESHOLDS.get("min_claim_cp", 8)

    def _sharpen(deltas: list[FeatureDelta]) -> list[FeatureDelta]:
        out: list[FeatureDelta] = []
        for d in deltas:
            s = series.get(d.name)
            if not s:
                out.append(d)
                continue
            direction = 1 if d.delta_cp > 0 else -1
            ext_idx = _feature_extremum_ply(s, direction, window)
            ext_val = s[ext_idx]
            if ext_idx != immediate_idx and direction * (ext_val - d.after_cp) >= threshold:
                out.append(
                    d.model_copy(
                        update={"after_cp": ext_val, "delta_cp": ext_val - d.before_cp}
                    )
                )
            else:
                out.append(d)
        out.sort(key=lambda x: -abs(x.delta_cp))
        return out

    return FeatureDiff(positive=_sharpen(diff.positive), negative=_sharpen(diff.negative))


THREAT_IDS = {"hanging_piece", "intent_targets_piece"}


def _threat_resolved(claim: Claim, boards: list[chess.Board], line: EnvisionedLine, support: int) -> str | None:
    """A threat that the displayed line makes irrelevant: the target is never taken,
    never moves away, and is no longer attacked at the end (its attacker was traded
    off or blocked). A target that is captured or retreats keeps the claim."""
    import re
    anchor = re.search(r"\b([a-h][1-8])\b", claim.text)
    if not anchor or claim.beneficiary not in ("white", "black"):
        return None
    square = chess.parse_square(anchor.group(1))
    piece = boards[support].piece_at(square)
    if piece is None:
        return None
    for index in range(support + 1, len(boards)):
        move = chess.Move.from_uci(line.line_uci[index - 1])
        if move.from_square == square or move.to_square == square:
            return None
    attacker = chess.WHITE if claim.beneficiary == "white" else chess.BLACK
    if boards[-1].is_attacked_by(attacker, square):
        return None
    return "threat_resolved_in_displayed_continuation"


def _two_horizon_claims(
    ctx: _FactsCtx, line: EnvisionedLine, eval_cp: int | None
) -> tuple[list[Claim], dict]:
    """Run the rules at two horizons and merge them:

    - *immediate* — start vs. the first quiet node after the move: what the move
      itself does (stated as fact);
    - *forecast* — start vs. the envisioned leaf: what only develops deeper in
      the line (hedged, tagged ``envisioned``).

    Immediate claims win per rule and per dominant feature; forecast claims are
    added only for features the move did not already move. Returns the merged
    claims and the leaf vector (for the material-loss fallback)."""
    leaf_vec = compute_feature_vector_fen(line.leaf_fen)
    if not replay(line):
        return [], leaf_vec
    if (
        not line.evidence_valid
        or not is_quiescent(ctx.board_before)
        or not is_quiescent(chess.Board(line.leaf_fen))
        or chess.Board(line.leaf_fen).is_game_over()
    ):
        return tactical_claims(line), leaf_vec
    immediate_fen = _first_quiet_fen(line)
    immediate_vec = compute_feature_vector_fen(immediate_fen)
    common = {
        "phase": ctx.phase_raw,
        "mover": ctx.mover.upper(),
        "eval_cp": eval_cp,
        "start_board": ctx.board_before,
        "cap": ctx.claim_policy != "contrastive",
    }
    # Numeric deltas, flags and square geometry must describe the SAME node.
    immediate_diff = diff_vectors(ctx.start_vec, immediate_vec)
    immediate = run_rules(
        immediate_diff,
        leaf_board=chess.Board(immediate_fen),
        **common,
    )
    forecast = run_rules(
        diff_vectors(ctx.start_vec, leaf_vec),
        leaf_board=chess.Board(line.leaf_fen),
        **common,
    )
    for claim in immediate:
        claim.support_ply = line.fens.index(immediate_fen) + 1
        claim.support_fen = immediate_fen
        claim.baseline_fen = line.start_fen
        claim.realization = "immediate" if claim.support_ply == 1 else "envisioned"
    for claim in forecast:
        claim.realization = "envisioned"
        claim.support_ply = len(line.fens)
        claim.support_fen = line.leaf_fen
        claim.baseline_fen = line.start_fen

    seen_rules = {(c.rule_id, c.beneficiary) for c in immediate}
    seen_feats = {
        (c.features_involved[0], c.beneficiary)
        for c in immediate
        if c.features_involved
    }

    def _is_new(c: Claim) -> bool:
        if (c.rule_id, c.beneficiary) in seen_rules:
            return False
        primary = c.features_involved[0] if c.features_involved else None
        return (primary, c.beneficiary) not in seen_feats

    # Deduplicate only after checking persistence: an unstable immediate
    # claim must not conceal a properly anchored, stable forecast fact.
    merged = immediate + forecast
    boards = replay(line)
    kept = []
    for claim in merged:
        reason = None
        support = claim.support_ply or 0
        if boards and support < len(boards) - 1:
            if "MATERIAL_BALANCE" in claim.features_involved:
                # Material standing is an exact state. A temporary capture
                # surplus must not survive a displayed recapture.
                from .material import _imbalance
                if any(_imbalance(b) != _imbalance(boards[support]) for b in boards[support + 1:]):
                    reason = "material_state_changes_in_displayed_continuation"
            elif any("PASS" in name for name in claim.features_involved):
                import re
                anchors = re.findall(r"\b[a-h][1-8]\b", claim.text)
                for anchor in anchors:
                    square = chess.parse_square(anchor)
                    piece = boards[support].piece_at(square)
                    if piece is None or piece.piece_type != chess.PAWN:
                        continue
                    # Follow the actual pawn, allowing legal advances/captures.
                    for index in range(support + 1, len(boards)):
                        move = chess.Move.from_uci(line.line_uci[index - 1])
                        if move.from_square == square and boards[index - 1].piece_at(square) == piece:
                            square = move.to_square
                        if boards[index].piece_at(square) != piece:
                            reason = "passed_pawn_anchor_removed_in_displayed_continuation"
                            break
            elif claim.rule_id in THREAT_IDS:
                reason = _threat_resolved(claim, boards, line, support)
        if reason:
            ctx.evidence_suppressions.append({"rule_id": claim.rule_id, "reason": reason,
                                              "support_ply": support, "branch": line.line_uci[:1]})
        else:
            kept.append(claim)
    immediate_ids = {id(c) for c in immediate}
    surviving_immediate = [c for c in kept if id(c) in immediate_ids]
    seen_rules = {(c.rule_id, c.beneficiary) for c in surviving_immediate}
    seen_feats = {(c.features_involved[0], c.beneficiary)
                  for c in surviving_immediate if c.features_involved}
    kept = surviving_immediate + [c for c in kept if id(c) not in immediate_ids and _is_new(c)]
    concrete = tactical_claims(line)
    if concrete:
        # Concrete actions explain the move before broad activity scores. The
        # opponent's incidental activity must not crowd out the reply evidence.
        activity = {"activity_improved", "activity_reduced", "rooks_activated", "king_under_pressure", "king_safer", "piece_pinned"}
        kept = [c for c in kept if c.rule_id not in activity]
    return concrete + kept, leaf_vec


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
        selection = {"version": "current_quiet_bounded", "selected_ply": len(played_line.fens)}
    claims, leaf_vec = _two_horizon_claims(ctx, played_line, move_event.eval_after_cp)
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


def _alternative_merits(
    best_claims: list[Claim], main_claims: list[Claim], mover: str, claim_policy: str = "current"
) -> list[Claim]:
    """The alternative's own merits (max 2), never repeating the main block's claims."""
    concrete = [c.model_copy(update={"is_concession": c.beneficiary not in (mover.lower(), None)})
                for c in _concrete_first(best_claims) if c.rule_id in CONCRETE_IDS][:3]
    if claim_policy == "contrastive":
        return concrete + [c for c in best_claims if c.rule_id not in CONCRETE_IDS]
    main_texts = {c.text for c in main_claims}
    mover_key = mover.lower()
    return concrete + [
        c
        for c in best_claims
        if c.rule_id not in CONCRETE_IDS and c.beneficiary in (mover_key, None) and c.text not in main_texts
    ][:max(0, MAX_ALTERNATIVE_MERITS - len(concrete))]


def _feature_advantages(
    played_diff: FeatureDiff | None, alt_diff: FeatureDiff
) -> list[FeatureAdvantage]:
    """Feature-diff-of-diffs (Improvement 3): the alternative's own
    start->leaf swing vs. the played move's, per feature, in cp. Additive to
    the existing claim-text-dedup ``claims`` list -- see
    ``envisioned.diff_of_diffs`` for the full rationale."""
    if played_diff is None:
        return []
    return diff_of_diffs(
        played_diff,
        alt_diff,
        min_abs_cp=THRESHOLDS.get("min_claim_cp", 8),
        max_features=MAX_FEATURE_ADVANTAGES,
    )


def _build_better_alternative(
    ctx: _FactsCtx, main_claims: list[Claim], played_diff: FeatureDiff | None = None
) -> BestAlternative | None:
    """The engine's preferred move + its merits, whenever a different move was
    played — so the better line can always be visualized, regardless of how
    small the gap is or whether it's the same piece (Guid). Uses the
    already-computed PVs, so it costs no extra engine search.
    """
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
    if ctx.horizon_policy == "adaptive_feature":
        best_line, _selection = select_adaptive_horizon(
            move_event.fen_before, best_pv_uci, root_eval_cp=best_cp,
            depth=ctx.depth, prober=ctx.prober,
        )
    else:
        best_line = envisioned_for_best_move(
            move_event.fen_before, best_pv_uci, best_eval_cp=best_cp, depth=ctx.depth
        )
        best_line = _trim_line(best_line, ctx.prober)
    best_claims, best_leaf_vec = _two_horizon_claims(ctx, best_line, best_cp)
    if _is_decided(best_cp, best_mate):
        best_claims = _material_only(best_claims)
    alt_diff = diff_vectors(ctx.start_vec, best_leaf_vec)
    return BestAlternative(
        san=move_event.best_move_san or move_event.best_move_uci,
        uci=move_event.best_move_uci,
        eval_cp=best_cp,
        eval_mate=best_mate,
        verdict=verdict_for_eval(best_cp, best_mate),
        display_line=best_line,
        claims=_alternative_merits(best_claims, main_claims, ctx.mover, ctx.claim_policy),
        feature_advantages=_feature_advantages(played_diff, alt_diff),
    )


def _build_inferior_alternative(
    ctx: _FactsCtx, main_claims: list[Claim], played_diff: FeatureDiff | None = None
) -> BestAlternative | None:
    """When the engine's best move was actually played, surface the runner-up
    (2nd-best PV) as a contrast for the "!" — so the reader sees what the move was
    chosen over (Guid). The wording later adapts to the gap: a clearly-worse
    runner-up reads as "weaker", a near-equal one as "a comparable alternative".
    Uses the already-computed PVs, so no extra search.
    """
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
    if ctx.horizon_policy == "adaptive_feature":
        alt_line, _selection = select_adaptive_horizon(
            move_event.fen_before, second_pv_uci, root_eval_cp=second_cp,
            depth=ctx.depth, prober=ctx.prober,
        )
    else:
        alt_line = envisioned_for_best_move(
            move_event.fen_before, second_pv_uci, best_eval_cp=second_cp, depth=ctx.depth
        )
        alt_line = _trim_line(alt_line, ctx.prober)
    alt_claims, alt_leaf_vec = _two_horizon_claims(ctx, alt_line, second_cp)
    if _is_decided(second_cp, second_mate):
        alt_claims = _material_only(alt_claims)
    alt_diff = diff_vectors(ctx.start_vec, alt_leaf_vec)
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
    # Decisive evals: positional claims are noise, but the material standing
    # ("White is a rook up") is exactly what explains the result — keep it.
    if _is_decided(eval_cp, eval_mate):
        before = claims
        claims = _material_only(claims)
        if len(claims) < len(before):
            logger.debug(
                "_filter_claims: decisive-position gate (eval_cp=%s, eval_mate=%s, "
                "threshold=%d) dropped %d non-material claim(s): %s",
                eval_cp,
                eval_mate,
                DECISIVE_CLAIM_CP,
                len(before) - len(claims),
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
                mover_pov_eval,
                MERIT_SUPPRESS_CP,
                len(before) - len(claims),
                [c.rule_id for c in before if c not in claims],
            )
    # When rules abstain, the verdict and legal variation are the explanation.
    # Do not turn a partial tactical leaf into an ungrounded material claim.
    return claims


def _attach_series(
    played_line: EnvisionedLine,
    better: BestAlternative | None,
    fen_before: str,
) -> tuple[EnvisionedLine, BestAlternative | None]:
    """Attach the per-ply feature series to the charted lines. Claim realization
    (immediate vs envisioned) is already set from the firing horizon."""
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

    ``prober`` is an optional cheap engine callable (FEN -> White-POV cp)
    used to sanity-check envisioned lines' leaves against a horizon-effect
    contradiction before any claim/feature-diff is computed from them
    (Improvement 1); omit it for engine-free/offline callers and tests."""
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
        ctx, claims, diff
    ) or _build_inferior_alternative(ctx, claims, diff))
    contrast_selection = None
    if better is not None:
        better = better.model_copy(
            update={"horizon_matched": horizons_match(played_line, better.display_line)}
        )
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
    claims = _concrete_first(claims)
    played_line, better = _attach_series(played_line, better, move_event.fen_before)
    verdict = verdict_for_transition(eval_before_cp, eval_cp, eval_mate, mover)
    if terminal_result:
        claims = []
        eval_mate = None  # integer zero cannot represent the winning side
        eval_cp = None
        verdict = (f"delivers checkmate for {mover}" if board_after.is_checkmate()
                   else "ends the game in a draw")

    return CommentFacts(
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
        },
    )
