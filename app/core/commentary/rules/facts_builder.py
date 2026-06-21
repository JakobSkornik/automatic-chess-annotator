"""CommentFacts assembly: the played line, its claims, the better alternative,
realization tagging, and the inviolable per-move fact record."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from app.core.commentary.features.envisioned import (
    diff_vectors,
    envisioned_for_best_move,
    envisioned_for_played_move,
)
from app.core.commentary.features.guid_features import compute_feature_vector_fen
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.comment_facts import (
    BestAlternative,
    Claim,
    CommentFacts,
    EnvisionedLine,
    FeatureDiff,
)

from .catalog import order_claims_for_mover, run_rules
from .constants import (
    _MATE_SCORE,
    DECISIVE_CLAIM_CP,
    EVAL_CONCESSION_CP,
    MATERIAL_LOSS_CP,
    MAX_ALTERNATIVE_MERITS,
    MERIT_SUPPRESS_CP,
)
from .realization import _annotate_realization, _line_feature_series
from .verdicts import verdict_for_eval, verdict_for_transition


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


def _played_line_claims(
    ctx: _FactsCtx,
) -> tuple[EnvisionedLine, FeatureDiff, dict, list[Claim]]:
    """Envisioned played line + its feature diff, leaf vector, and ordered claims."""
    move_event = ctx.move_event
    engine_meta = (ctx.row.hidden_features or {}).get("_engine") or {}
    after_pv = list(engine_meta.get("after_pv_uci") or [])
    played_line = envisioned_for_played_move(
        move_event.fen_before,
        move_event.uci,
        after_pv,
        played_eval_cp=move_event.eval_after_cp,
        depth=ctx.depth,
    )
    leaf_vec = compute_feature_vector_fen(played_line.leaf_fen)
    diff = diff_vectors(ctx.start_vec, leaf_vec)
    claims = run_rules(
        diff,
        phase=ctx.phase_raw,
        mover=ctx.mover.upper(),
        eval_cp=move_event.eval_after_cp,
        start_board=ctx.board_before,
        leaf_board=chess.Board(played_line.leaf_fen),
    )
    return played_line, diff, leaf_vec, order_claims_for_mover(claims, ctx.mover)


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
    best_claims: list[Claim], main_claims: list[Claim], mover: str
) -> list[Claim]:
    """The alternative's own merits (max 2), never repeating the main block's claims."""
    main_texts = {c.text for c in main_claims}
    mover_key = mover.lower()
    return [
        c
        for c in best_claims
        if c.beneficiary in (mover_key, None) and c.text not in main_texts
    ][:MAX_ALTERNATIVE_MERITS]


def _build_better_alternative(
    ctx: _FactsCtx, main_claims: list[Claim]
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
    best_line = envisioned_for_best_move(
        move_event.fen_before, best_pv_uci, best_eval_cp=best_cp, depth=ctx.depth
    )
    best_diff = diff_vectors(
        ctx.start_vec, compute_feature_vector_fen(best_line.leaf_fen)
    )
    best_claims = run_rules(
        best_diff,
        phase=ctx.phase_raw,
        mover=ctx.mover.upper(),
        eval_cp=best_cp,
        start_board=ctx.board_before,
        leaf_board=chess.Board(best_line.leaf_fen),
    )
    return BestAlternative(
        san=move_event.best_move_san or move_event.best_move_uci,
        uci=move_event.best_move_uci,
        eval_cp=best_cp,
        verdict=verdict_for_eval(best_cp, best_mate),
        display_line=best_line,
        claims=_alternative_merits(best_claims, main_claims, ctx.mover),
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


def _concession_mode(move_quality: str) -> str:
    """For dubious moves opponent-favoring claims explain the error, not trade-offs."""
    return "consequence" if move_quality in _DUBIOUS else "tradeoff"


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
    # Decisive evals: positional claims are noise (a passed pawn in mate-in-4).
    if eval_mate is not None or (
        eval_cp is not None and abs(eval_cp) > DECISIVE_CLAIM_CP
    ):
        claims = []
    mover_pov_eval = (
        None if eval_cp is None else (eval_cp if mover == "White" else -eval_cp)
    )
    if mover_pov_eval is not None and mover_pov_eval <= -MERIT_SUPPRESS_CP:
        mover_key = mover.lower()
        claims = [
            c
            for c in claims
            if c.is_concession or c.beneficiary not in (mover_key, None)
        ]
    if not claims and move_quality in _DUBIOUS:
        fallback = _fallback_claim(ctx, leaf_vec, refutation_san)
        if fallback is not None:
            claims = [fallback]
    return claims


def _attach_series_and_realization(
    played_line: EnvisionedLine,
    claims: list[Claim],
    better: BestAlternative | None,
    fen_before: str,
) -> tuple[EnvisionedLine, BestAlternative | None]:
    """Attach the per-ply feature series and tag each claim immediate/envisioned."""
    played_line = _with_feature_series(played_line, fen_before)
    _annotate_realization(claims, played_line.feature_series)
    if better is not None and better.display_line is not None:
        better = better.model_copy(
            update={
                "display_line": _with_feature_series(better.display_line, fen_before)
            }
        )
        _annotate_realization(better.claims, better.display_line.feature_series)
    return played_line, better


def build_comment_facts(
    row: AnalyzedMoveData,
    move_event: MoveEvent,
    *,
    depth: int = 16,
) -> CommentFacts | None:
    """Assemble the move's inviolable facts from data the engine pass already paid for."""
    phase_raw = row.phase_raw or "mid"
    if phase_raw == "early":
        return None

    board_before = chess.Board(move_event.fen_before)
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
    )

    played_line, diff, leaf_vec, claims = _played_line_claims(ctx)
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
    better = _build_better_alternative(ctx, claims)
    played_line, better = _attach_series_and_realization(
        played_line, claims, better, move_event.fen_before
    )

    return CommentFacts(
        ply=move_event.ply,
        san=move_event.san,
        uci=move_event.uci,
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
        concession_mode=_concession_mode(move_quality),
    )
