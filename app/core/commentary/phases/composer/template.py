"""Deterministic Guid-format template: the no-LLM rendering, and the fallback
used whenever an LLM rendering fails the fact contract."""

from __future__ import annotations

from app.core.commentary.rules.constants import INFERIOR_ALT_WEAKER_CP
from app.models.comment_facts import Claim, CommentFacts

from .framing import (
    _ARCHETYPE_OPENER,
    _is_alt_materially_better,
    _missed_noun,
    alt_gap_cp,
)
from .tokens import _alt_pv_token, _gerundize, _move_label, eval_token, pv_token

LONG_LINE_PLIES = 6  # lines this long describe the envisioned position (state form)


def _claim_text(c: Claim, *, prefer_state: bool) -> str:
    if prefer_state and c.text_state:
        return c.text_state
    return c.text


def _render_head(facts: CommentFacts, variant: int, *, archetype: str | None) -> str:
    pv, ev, move = pv_token(facts), eval_token(facts), _move_label(facts)
    opener = _ARCHETYPE_OPENER.get(archetype or "")
    head = f"{move} {opener} {facts.verdict}" if opener else f"{move} {facts.verdict}"
    if pv:
        head += f" after {pv}" if variant == 0 else f": {pv}"
    if ev:
        head += f" {ev}"
    return head + "."


def _render_concessions(
    concessions: list[Claim], concession_mode: str, variant: int, prefer_state: bool
) -> list[str]:
    """Immediate concessions stated as fact; envisioned ones framed as a risk."""
    if not concessions:
        return []
    immediate = [c for c in concessions if c.realization != "envisioned"]
    envisioned = [c for c in concessions if c.realization == "envisioned"]
    out: list[str] = []
    if immediate:
        if concession_mode == "consequence":
            prefix = "Now " if variant == 0 else "The drawback: "
            clauses = [c.text.rstrip(".") for c in immediate]
        else:
            prefix = "In return, " if variant == 0 else "On the other hand, "
            clauses = [
                _claim_text(c, prefer_state=prefer_state).rstrip(".") for c in immediate
            ]
        out.append(prefix + " and ".join(clauses) + ".")
    if envisioned:
        clauses = [c.text.rstrip(".") for c in envisioned]
        out.append("Down the line, " + " and ".join(clauses) + ".")
    return out


def _render_claims(facts: CommentFacts, variant: int) -> list[str]:
    if not facts.claims:
        return []
    # Long quiescent lines describe the envisioned position -> state form.
    prefer_state = bool(
        facts.display_line and len(facts.display_line.line_san) >= LONG_LINE_PLIES
    )
    merits = [c for c in facts.claims if not c.is_concession]
    concessions = [c for c in facts.claims if c.is_concession]
    out: list[str] = []
    if merits:
        out.append(" ".join(_claim_text(c, prefer_state=prefer_state) for c in merits))
    out += _render_concessions(
        concessions, facts.concession_mode, variant, prefer_state
    )
    return out


def _render_alternative(facts: CommentFacts, variant: int) -> str | None:
    alt = facts.better_alternative
    if alt is None:
        return None
    alt_pv = _alt_pv_token(alt)
    if alt.is_inferior:
        # The played move WAS best; contrast it with the runner-up. Wording
        # depends on the gap: clearly worse -> "Weaker was", else "comparable".
        gap = alt_gap_cp(facts)  # mover-POV; <= 0 for a runner-up
        clearly_weaker = gap is None or -gap >= INFERIOR_ALT_WEAKER_CP
        lead = (
            f"Weaker was {alt.san}"
            if clearly_weaker
            else (f"A comparable alternative was {alt.san}")
        )
        s = lead
        if alt.verdict:
            s += f", which {alt.verdict}"
        if alt_pv:
            s += f" after {alt_pv}"
        s += "."
        if alt.claims:
            s += " " + " ".join(c.text for c in alt.claims)
        return s
    if not _is_alt_materially_better(facts):
        # Roughly equal: present it as an option, not a miss.
        s = f"A comparable alternative was {alt.san}"
        if alt_pv:
            s += f" after {alt_pv}"
        s += "."
        if alt.claims:
            s += " The point: " + " ".join(c.text for c in alt.claims)
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
    if alt.claims:
        # The alternative's merits are what the mover passed up.
        noun = _missed_noun(alt.claims)
        s += f" {facts.mover} missed the {noun} here — " + " ".join(
            c.text for c in alt.claims
        )
    return s


def render_facts_template(facts: CommentFacts, *, archetype: str | None = None) -> str:
    """Guid-format rendering with deterministic phrasing rotation (per ply), so
    the pattern does not repeat verbatim move after move."""
    variant = facts.ply % 2
    parts = [_render_head(facts, variant, archetype=archetype)]
    if facts.refutation_san:
        parts.append(f"The move is punished by {facts.refutation_san}.")
    parts += _render_claims(facts, variant)
    alt = _render_alternative(facts, variant)
    if alt:
        parts.append(alt)
    return " ".join(p for p in parts if p).strip()
