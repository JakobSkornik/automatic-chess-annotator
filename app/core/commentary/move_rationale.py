"""Build MoveRationale from engine features (no LLM)."""

from __future__ import annotations

import chess

from app.core.commentary.motif_phrases import glossary_phrase_for
from app.models.chess_events import FutureLineDelta, MoveEvent, MoveRationale


def _squares_attacked_by_moved_piece(board: chess.Board, to_square: int) -> str:
    try:
        names = []
        for sq in chess.SQUARES:
            if sq in board.attacks(to_square):
                names.append(chess.square_name(sq))
        return ", ".join(names[:12]) + ("..." if len(names) > 12 else "")
    except Exception:
        return ""


def build_rationale(
    me: MoveEvent,
    future_delta: FutureLineDelta | None = None,
) -> MoveRationale:
    """Symbolic 'why this move matters' for composer prompts."""
    swing = int(me.eval_swing_cp) if me.eval_swing_cp is not None else 0

    tactical_keys = [m.value for m in me.tactical_motifs]
    strategic_keys = [m.value for m in me.strategic_motifs]
    glossary_phrasings = {k: glossary_phrase_for(k) for k in tactical_keys + strategic_keys}

    immediate = ""
    narrative_template = ""
    try:
        b = chess.Board(me.fen_before)
        m = chess.Move.from_uci(me.uci)
        cap = b.piece_at(m.to_square) if b.is_capture(m) else None
        movers = b.piece_at(m.from_square)
        mover_sym = movers.symbol().upper() if movers else "?"
        dest = chess.square_name(m.to_square)
        if b.is_capture(m):
            vs = cap.symbol() if cap else "?"
            immediate = f"Captures {vs} on {dest} with the {mover_sym}."
            narrative_template = immediate
        elif b.gives_check(m):
            immediate = f"Check with the {mover_sym} to {dest}."
            narrative_template = immediate
        elif me.tactical_motifs:
            immediate = f"Tactical theme: {me.tactical_motifs[0].value.replace('_', ' ')}."
            narrative_template = immediate
        elif me.strategic_motifs:
            immediate = f"Strategic idea: {me.strategic_motifs[0].value.replace('_', ' ')}."
            narrative_template = immediate
        else:
            ba = chess.Board(me.fen_after)
            att = _squares_attacked_by_moved_piece(ba, m.to_square)
            immediate = (
                f"Piece move {mover_sym} to {dest}; influences squares including {att}."
                if att
                else f"Quiet repositioning of the {mover_sym} to {dest}."
            )
            narrative_template = immediate
    except Exception:
        immediate = "Position update."
        narrative_template = immediate

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

    primary = primary_motif_label(me)

    counterfactual: str | None = None
    if me.best_move_san and me.uci != me.best_move_uci:
        counterfactual = (
            f"If {me.best_move_san} instead, the engine-preferred continuation starts "
            f"(fits the plan pressure better when the game arc calls for it)."
        )
    if future_delta and future_delta.best_line_san:
        counterfactual = (counterfactual or "") + f" Sample best line: {' '.join(future_delta.best_line_san[:5])}."

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

    coach_scratchpad = {
        "why": immediate.strip(),
        "risk": (risk or "").strip(),
        "plan": f"Played: {played_plan or 'n/a'} | Engine lean: {best_plan or 'n/a'}".strip(),
        "counterplay": (counterfactual or "").strip(),
    }

    return MoveRationale(
        eval_change_cp=swing,
        immediate_effect=immediate.strip(),
        future_effect=future_eff.strip(),
        motif=motif,
        counterfactual=counterfactual.strip() if counterfactual else None,
        played_plan=played_plan,
        best_plan=best_plan,
        risk=risk,
        glossary_phrasings=glossary_phrasings,
        primary_motif_label=primary,
        narrative_template=narrative_template.strip(),
        coach_scratchpad=coach_scratchpad,
    )


def primary_motif_label(me: MoveEvent) -> str:
    if me.tactical_motifs:
        return me.tactical_motifs[0].value
    if me.strategic_motifs:
        return me.strategic_motifs[0].value
    return "none"
