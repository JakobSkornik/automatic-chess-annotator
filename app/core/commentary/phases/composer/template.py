"""Deterministic Guid-format template: the no-LLM rendering, and the fallback
used whenever an LLM rendering fails the fact contract."""

from __future__ import annotations

import chess

from app.core.commentary.rules.constants import INFERIOR_ALT_WEAKER_CP
from app.models.comment_facts import Claim, CommentFacts

from .framing import (
    _ARCHETYPE_OPENER,
    _is_alt_materially_better,
    _missed_noun,
    alt_gap_cp,
    alternative_comparison,
)
from .tokens import _alt_pv_token, _gerundize, _move_label, eval_token, pv_token


def _claim_text(c: Claim, *, prefer_state: bool) -> str:
    """Change-form vs state-form (Guid subproblem 5), per claim: a change the
    move itself makes is stated in change form; one that only develops deeper
    in the line reads as a property of the envisioned position."""
    if prefer_state and c.realization != "immediate" and c.text_state:
        return c.text_state
    return c.text


def _anchored_claims(claims, line) -> str:
    """Keep later observations tied to their own displayed supporting move."""
    out = []
    for c in claims:
        anchor = None
        if line and c.support_ply and 1 <= c.support_ply <= len(line.line_uci):
            try:
                board = chess.Board(line.start_fen)
                for uci in line.line_uci[:c.support_ply]:
                    move = chess.Move.from_uci(uci)
                    anchor = f"{board.fullmove_number}{'.' if board.turn else '...'}{board.san(move)}"
                    board.push(move)
            except ValueError:
                anchor = None
        prefix = f"After {anchor} in the analysed line, " if anchor else "In the analysed continuation, "
        text = _claim_text(c, prefer_state=True)
        if text.lower().startswith("in the displayed exchange"):
            # This sentence already identifies its own displayed exchange.
            prefix = ""
        out.append(prefix + text)
    return " ".join(out)


def _render_head(facts: CommentFacts, variant: int, *, archetype: str | None) -> str:
    pv, ev, move = pv_token(facts), eval_token(facts), _move_label(facts)
    opener = _ARCHETYPE_OPENER.get(archetype or "")
    head = f"{move} {opener} {facts.verdict}" if opener else f"{move} {facts.verdict}"
    if pv:
        # Rotation: "after {pv}" / ": {pv}" / " ({pv})" — three shapes so the
        # Guid head does not repeat verbatim between adjacent comments.
        if variant == 0:
            head += f" after {pv}"
        elif variant == 1:
            head += f": {pv}"
        else:
            head += f" ({pv})"
    if ev:
        head += f" {ev}"
    return head + "."


def _render_concessions(
    concessions: list[Claim], concession_mode: str, variant: int, line=None
) -> list[str]:
    """Immediate concessions stated as fact; envisioned ones framed as a risk."""
    if not concessions:
        return []
    immediate = [c for c in concessions if c.realization != "envisioned"]
    envisioned = [c for c in concessions if c.realization == "envisioned"]
    out: list[str] = []
    if immediate:
        if concession_mode == "consequence":
            prefixes = ("Now ", "The drawback: ", "The problem: ")
            prefix = prefixes[variant % len(prefixes)]
            clauses = [c.text.rstrip(".") for c in immediate]
        else:
            prefixes = (
                "In return, ",
                "On the other hand, ",
                "The trade-off is that ",
            )
            prefix = prefixes[variant % len(prefixes)]
            clauses = [_claim_text(c, prefer_state=False).rstrip(".") for c in immediate]
        out.append(prefix + " and ".join(clauses) + ".")
    if envisioned:
        # Envisioned concessions read as the state the line is heading toward.
        out.append(_anchored_claims(envisioned, line))
    return out


def _render_claims(facts: CommentFacts, variant: int) -> list[str]:
    if not facts.claims:
        return []
    merits = [c for c in facts.claims if not c.is_concession]
    concessions = [c for c in facts.claims if c.is_concession]
    out: list[str] = []
    # Immediate merits are facts; envisioned merits are only set up by the
    # move, so they must be hedged like envisioned concessions (the LLM
    # prompt requires the same — the template must match it).
    immediate_merits = [c for c in merits if c.realization != "envisioned"]
    envisioned_merits = [c for c in merits if c.realization == "envisioned"]
    if immediate_merits:
        out.append(
            " ".join(_claim_text(c, prefer_state=True) for c in immediate_merits)
        )
    out += _render_concessions(
        [c for c in concessions if c.realization != "envisioned"],
        facts.concession_mode, variant, facts.display_line)
    forecasts = sorted(
        envisioned_merits + [c for c in concessions if c.realization == "envisioned"],
        key=lambda c: c.support_ply or 0)
    if forecasts:
        out.append(_anchored_claims(forecasts, facts.display_line))
    return out


def _render_alternative(facts: CommentFacts, variant: int) -> str | None:
    alt = facts.better_alternative
    if alt is None:
        return None
    def explanation() -> str:
        immediate = [c.text for c in alt.claims if c.realization == "immediate"]
        forecast = [c for c in alt.claims if c.realization != "immediate"]
        return " ".join(immediate + ([_anchored_claims(forecast, alt.display_line)] if forecast else []))
    alt_pv = _alt_pv_token(alt)
    if alt.is_inferior:
        # The played move WAS best; contrast it with the runner-up. Wording
        # depends on the gap: clearly worse -> "Weaker was", else "comparable".
        comparison = alternative_comparison(facts, threshold_cp=INFERIOR_ALT_WEAKER_CP)
        clearly_weaker = comparison == "weaker"
        lead = (
            f"Weaker was {alt.san}"
            if clearly_weaker
            else (f"A comparable alternative was {alt.san}" if comparison == "comparable"
                  else f"Another alternative was {alt.san}")
        )
        s = lead
        if alt.verdict:
            s += f", which {alt.verdict}"
        if alt_pv:
            s += f" after {alt_pv}"
        s += "."
        if alt.claims:
            s += " " + explanation()
        return s
    if not _is_alt_materially_better(facts):
        # Roughly equal: present it as an option, not a miss.
        comparison = alternative_comparison(facts)
        label = "A comparable alternative" if comparison == "comparable" else (
            "A weaker alternative" if comparison == "weaker" else "Another alternative")
        s = f"{label} was {alt.san}"
        if alt.verdict:
            s += f", which {alt.verdict}"
        if alt_pv:
            s += f" after {alt_pv}"
        s += "."
        if alt.claims:
            s += " " + explanation()
        return s
    if variant == 0:
        s = f"Better was {alt.san}"
        if alt.verdict:
            s += f", which {alt.verdict}"
        if alt_pv:
            s += f" after {alt_pv}"
    else:
        s = f"A better move was {alt.san}"
        if alt.verdict:
            s += f", {_gerundize(alt.verdict)}"
        if alt_pv:
            s += f" after {alt_pv}"
    s += "."
    gap = alt_gap_cp(facts)
    if gap is not None:
        s += (f" {facts.mover} missed a better continuation"
              f" ({gap / 100:.2f} pawns better in the engine evaluation).")
    if alt.claims:
        s += " " + explanation()
    return s


def render_facts_template(facts: CommentFacts, *, archetype: str | None = None) -> str:
    """Guid-format rendering with deterministic phrasing rotation (per ply), so
    the pattern does not repeat verbatim move after move."""
    variant = facts.ply % 3
    parts = [_render_head(facts, variant, archetype=archetype)]
    if facts.refutation_san:
        parts.append(f"The move is punished by {facts.refutation_san}.")
    parts += _render_claims(facts, variant)
    alt = _render_alternative(facts, variant)
    if alt:
        parts.append(alt)
    return " ".join(p for p in parts if p).strip()
