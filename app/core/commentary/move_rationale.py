"""Build MoveRationale from engine features (no LLM)."""

from __future__ import annotations

import chess

from app.models.chess_events import FutureLineDelta, MoveEvent, MoveRationale


def build_rationale(
    me: MoveEvent,
    future_delta: FutureLineDelta | None = None,
) -> MoveRationale:
    """Symbolic 'why this move matters' for composer prompts."""
    swing = int(me.eval_swing_cp) if me.eval_swing_cp is not None else 0

    immediate = ""
    try:
        b = chess.Board(me.fen_before)
        m = chess.Move.from_uci(me.uci)
        if b.is_capture(m):
            immediate = "Capture."
        elif b.gives_check(m):
            immediate = "Check."
        elif me.tactical_motifs:
            immediate = f"Tactical idea: {me.tactical_motifs[0].value}."
        else:
            immediate = "Quiet repositioning / structural change."
    except Exception:
        immediate = "Position update."

    future_eff = ""
    if future_delta and future_delta.feature_deltas:
        top = sorted(
            future_delta.feature_deltas.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:3]
        future_eff = "; ".join(f"{k} {v:+.2f}" for k, v in top)
    elif future_delta and future_delta.eval_gap_cp is not None:
        future_eff = f"Leaf eval gap (best−played) ≈ {future_delta.eval_gap_cp} cp"
    else:
        future_eff = "Future line not deep-analysed for this move."

    motif: str | None = None
    if me.tactical_motifs:
        motif = me.tactical_motifs[0].value
    elif me.strategic_motifs:
        motif = me.strategic_motifs[0].value

    counterfactual: str | None = None
    if me.best_move_san and me.uci != me.best_move_uci:
        counterfactual = f"If {me.best_move_san} instead, the engine-preferred continuation starts."
    if future_delta and future_delta.best_line_san:
        counterfactual = (counterfactual or "") + f" Best line sample: {' '.join(future_delta.best_line_san[:5])}."

    played_plan = None
    best_plan = None
    if me.plan_comparison:
        pc = me.plan_comparison
        if pc.played_plan_seed:
            played_plan = pc.played_plan_seed
        if pc.best_plan_seed:
            best_plan = pc.best_plan_seed

    risk = None
    if me.key_moment_type:
        risk = f"Key moment: {me.key_moment_type}"

    return MoveRationale(
        eval_change_cp=swing,
        immediate_effect=immediate.strip(),
        future_effect=future_eff.strip(),
        motif=motif,
        counterfactual=counterfactual.strip() if counterfactual else None,
        played_plan=played_plan,
        best_plan=best_plan,
        risk=risk,
    )


def primary_motif_label(me: MoveEvent) -> str:
    if me.tactical_motifs:
        return me.tactical_motifs[0].value
    if me.strategic_motifs:
        return me.strategic_motifs[0].value
    return "none"
