"""Deterministic Guid-format template: the no-LLM rendering, and the fallback
used whenever an LLM rendering fails the fact contract."""

from __future__ import annotations

import re

import chess

from app.core.commentary import pipeline_flags
from app.core.commentary.rules.tactical_evidence import CONCRETE_IDS
from app.core.commentary.rules.verdicts import EQUAL_MAX_CP
from app.core.commentary.rules.constants import INFERIOR_ALT_WEAKER_CP
from app.models.comment_facts import Claim, CommentFacts

from .framing import other_moves_concede
from .framing import (
    _ARCHETYPE_OPENER,
    _is_alt_materially_better,
    _missed_noun,
    alt_gap_cp,
    alternative_comparison,
)
from .tokens import (
    _alt_pv_token, _gerundize, _move_label, alternative_label, eval_token, lower_start, pv_token, reply_label,
)


def _plain() -> bool:
    """v19 template_without_lines: no variation, no numbers, lower case after a lead-in."""
    return pipeline_flags.enabled("template_without_lines")


def _joined(text: str) -> str:
    return lower_start(text) if _plain() else text


def _tidy() -> bool:
    """v19 statement_wording: the template's own part of it."""
    return pipeline_flags.enabled("statement_wording")


def _tidied(claims: list[Claim]) -> list[Claim]:
    """Leave out a statement that another one already makes: the material count in the middle
    of an exchange (the recapture follows at once), and the capture of a piece that the move
    is already said to have left unsafe."""
    captures = {c.support_ply for c in claims if c.rule_id == "legal_capture"}
    unsafe = {c.entity.split(":")[-1] for c in claims if c.rule_id == "piece_left_en_prise" and c.entity}
    out = []
    for c in claims:
        if (c.rule_id.startswith("material") and c.persistence == "transient"
                and (c.support_ply or 0) + 1 in captures):
            continue
        if c.rule_id == "legal_capture" and c.entity and c.entity.split(":")[-1] in unsafe:
            continue
        out.append(c)
    return out


_WHY_HOLE = re.compile(r": no (?:White|Black) pawn can attack it any more")


def _explained_once(text: str) -> str:
    """What a hole is is explained at the first hole of a comment only."""
    seen = []

    def keep(match):
        seen.append(match)
        return match[0] if len(seen) == 1 else ""

    return _WHY_HOLE.sub(keep, text)


def _first_ply(c: Claim, line) -> int:
    """The move of the line after which the statement first holds.

    The facts compare features at the first quiet position, so everything inside a forcing
    stretch carries that position's ply and is marked immediate (43...c6 44.Rxh6 Kb5: the lost
    pawn and the king step both carry ply 3). v19 statement_wording walks the per-move feature
    series back to the first move after which the statement's features already had their
    value: the pawn is lost after 44.Rxh6, the king steps up after 44...Kb5, and what the
    played move itself changed stays with the played move. Without a feature series: a
    tactical statement and an intention keep their own ply (it is exact, or the statement was
    read off the position at that ply); any other statement stays with the played move, as
    before."""
    last = c.support_ply or 1
    if not _tidy() or last <= 1 or c.realization == "envisioned" or c.supplementary:
        return last
    if c.rule_id == "exchange_sequence":
        return 1  # the exchange starts with the played move
    series = getattr(line, "feature_series", None) or {}
    first = None
    for name in c.features_involved:
        values = series.get(name)
        if not values or len(values) <= last:
            continue
        k = last
        while k > 0 and values[k - 1] == values[last]:
            k -= 1
        first = max(first or 1, k)
    if first is None:
        exact = c.rule_id in CONCRETE_IDS or c.rule_id.startswith("intent_") or not c.features_involved
        return last if exact else 1
    return first


def _after_first(c: Claim, line) -> bool:
    """v19 statement_wording: a statement that first holds after a later move of the line
    names that move, also when the facts mark it as immediate."""
    return _tidy() and not c.supplementary and _first_ply(c, line) > 1


def _listed(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


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
        ply = _first_ply(c, line) if c.support_ply else None
        if line and ply and 1 <= ply <= len(line.line_uci):
            try:
                board = chess.Board(line.start_fen)
                for uci in line.line_uci[:ply]:
                    move = chess.Move.from_uci(uci)
                    anchor = f"{board.fullmove_number}{'.' if board.turn else '...'}{board.san(move)}"
                    board.push(move)
            except ValueError:
                anchor = None
        prefix = f"After {anchor} in the analysed line, " if anchor else "In the analysed continuation, "
        text = _claim_text(c, prefer_state=True)
        if text.lower().startswith("in the displayed exchange") or c.rule_id == "exchange_sequence":
            # This sentence already identifies its own displayed exchange.
            prefix = ""
        if _plain():
            # Statements resting on the same move share one lead-in.
            if prefix and out and out[-1][0] == prefix:
                out[-1][1].append(text)
            else:
                out.append((prefix, [text]))
        else:
            out.append(prefix + text)
    if not _plain():
        return " ".join(out)
    sentences, lead_ins = [], 0
    for prefix, texts in out:
        if not prefix:
            sentences += texts
            continue
        lead_ins += 1
        if lead_ins > 1:  # "in the analysed line" is said once
            prefix = prefix.replace(" in the analysed line", "")
        parts = [lower_start(t.rstrip(".")) for t in texts]
        if _tidy():
            # A statement that carries its own explanation (after a colon) ends its sentence.
            explained = [p for p in parts if ":" in p]
            groups = [[p for p in parts if ":" not in p] + explained[:1]] + [[p] for p in explained[1:]]
            sentences += [prefix + _listed(group) + "." for group in groups if group]
            continue
        sentences.append(prefix + _listed(parts) + ".")
    return " ".join(sentences)


def _render_head(facts: CommentFacts, variant: int, *, archetype: str | None) -> str:
    pv, ev, move = pv_token(facts), eval_token(facts), _move_label(facts)
    opener = _ARCHETYPE_OPENER.get(archetype or "")
    head = f"{move} {opener} {facts.verdict}" if opener else f"{move} {facts.verdict}"
    if _plain():
        return head + ("." if facts.terminal_result is None else f" {ev}.")
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
            clauses = [_joined(c.text.rstrip(".")) for c in immediate]
        else:
            prefixes = (
                "In return, ",
                "On the other hand, ",
                "The trade-off is that ",
            )
            prefix = prefixes[variant % len(prefixes)]
            clauses = [_joined(_claim_text(c, prefer_state=False).rstrip(".")) for c in immediate]
        if _tidy():
            # At most two statements in a sentence; one with its own explanation ends it.
            groups, current = [], []
            for clause in clauses:
                current.append(clause)
                if ":" in clause or len(current) == 2:
                    groups.append(current)
                    current = []
            if current:
                groups.append(current)
            for index, group in enumerate(groups):
                # No lead-in before a statement that brings its own "After ...", and no second colon.
                own = group[0].lower().startswith("after ") or (":" in prefix and ":" in group[0])
                lead = prefix if index == 0 and not own else ""
                sentence = lead + " and ".join(group) + "."
                out.append(sentence[0].upper() + sentence[1:])
        else:
            out.append(prefix + " and ".join(clauses) + ".")
    if envisioned:
        # Envisioned concessions read as the state the line is heading toward.
        out.append(_anchored_claims(envisioned, line))
    return out


def _render_claims(facts: CommentFacts, variant: int) -> list[str]:
    if not facts.claims:
        return []
    claims = _tidied(facts.claims) if _tidy() else facts.claims
    merits = [c for c in claims if not c.is_concession]
    concessions = [c for c in claims if c.is_concession]
    out: list[str] = []
    # Immediate merits are facts; envisioned merits are only set up by the
    # move, so they must be hedged like envisioned concessions (the LLM
    # prompt requires the same — the template must match it).
    line = facts.display_line
    immediate_merits = [c for c in merits if c.realization != "envisioned" and not _after_first(c, line)]
    envisioned_merits = [c for c in merits if c.realization == "envisioned" or _after_first(c, line)]
    if immediate_merits:
        out.append(
            " ".join(_claim_text(c, prefer_state=True) for c in immediate_merits)
        )
    out += _render_concessions(
        [c for c in concessions if c.realization != "envisioned" and not _after_first(c, line)],
        facts.concession_mode, variant, facts.display_line)
    forecasts = sorted(
        envisioned_merits + [c for c in concessions if c.realization == "envisioned" or _after_first(c, line)],
        key=lambda c: (_first_ply(c, line) if c.support_ply else 0, _tidy() and c.rule_id != "legal_capture"))
    if forecasts:
        out.append(_anchored_claims(forecasts, facts.display_line))
    return out


def _render_alternative(facts: CommentFacts, variant: int) -> str | None:
    alt = facts.better_alternative
    if alt is None:
        return None
    def explanation() -> str:
        line = alt.display_line
        # "The lines part at ..." only says that the two lines differ; the template leaves it out.
        claims = [c for c in alt.claims if not (_tidy() and c.rule_id == "lines_diverge")]
        immediate = [c.text for c in claims if c.realization == "immediate" and not _after_first(c, line)]
        forecast = [c for c in claims if c.realization != "immediate" or _after_first(c, line)]
        if _tidy():
            forecast.sort(key=lambda c: _first_ply(c, line) if c.support_ply else 0)
        return " ".join(immediate + ([_anchored_claims(forecast, alt.display_line)] if forecast else []))
    alt_pv = "" if _plain() else _alt_pv_token(alt)
    # v19 comment_blocks: the alternative is named with its move number.
    alt_san = alternative_label(facts) if pipeline_flags.enabled("comment_blocks") else alt.san
    if alt.is_inferior:
        # The played move WAS best; contrast it with the runner-up. Wording
        # depends on the gap: clearly worse -> "Weaker was", else "comparable".
        comparison = alternative_comparison(facts, threshold_cp=INFERIOR_ALT_WEAKER_CP)
        clearly_weaker = comparison == "weaker"
        if clearly_weaker and pipeline_flags.enabled("other_moves_concede"):
            # A much worse second choice is not named or explained.
            conceded = other_moves_concede(facts)
            return f"Other moves concede {_listed(conceded)}." if conceded else "Other moves are clearly weaker."
        lead = (
            f"Weaker was {alt_san}"
            if clearly_weaker
            else (f"A comparable alternative was {alt_san}" if comparison == "comparable"
                  else f"Another alternative was {alt_san}")
        )
        s = lead
        if alt.verdict:
            # A weaker move that keeps the mover on top: "which still ...".
            sign = 1 if facts.mover == "White" else -1
            still = (_tidy() and clearly_weaker and alt.eval_mate is None and alt.eval_cp is not None
                     and sign * alt.eval_cp >= EQUAL_MAX_CP)
            s += f", which {'still ' if still else ''}{alt.verdict}"
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
        s = f"{label} was {alt_san}"
        if alt.verdict:
            s += f", which {alt.verdict}"
        if alt_pv:
            s += f" after {alt_pv}"
        s += "."
        if alt.claims:
            s += " " + explanation()
        return s
    if variant == 0:
        s = f"Better was {alt_san}"
        if alt.verdict:
            s += f", which {alt.verdict}"
        if alt_pv:
            s += f" after {alt_pv}"
    else:
        s = f"A better move was {alt_san}"
        if alt.verdict:
            s += f", {_gerundize(alt.verdict)}"
        if alt_pv:
            s += f" after {alt_pv}"
    s += "."
    gap = alt_gap_cp(facts)
    if gap is not None and not (_tidy() and _plain()):  # "A better move was" already says it
        s += f" {facts.mover} missed a better continuation" + (
            "." if _plain() else f" ({gap / 100:.2f} pawns better in the engine evaluation).")
    if alt.claims:
        s += " " + explanation()
    return s


def render_facts_template(facts: CommentFacts, *, archetype: str | None = None) -> str:
    """Guid-format rendering with deterministic phrasing rotation (per ply), so
    the pattern does not repeat verbatim move after move."""
    variant = facts.ply % 3
    parts = [_render_head(facts, variant, archetype=archetype)]
    blocks = pipeline_flags.enabled("comment_blocks")
    if facts.refutation_san:
        reply = reply_label(facts, facts.refutation_san) if blocks else facts.refutation_san
        claim_parts = _render_claims(facts, variant)
        lead = f"After {reply} in the analysed line, "
        merged = False
        if _tidy() and _plain():
            # The first statement about the reply says how it punishes: one sentence, not two.
            for index, part in enumerate(claim_parts):
                if part.startswith(lead) and ":" not in part[len(lead):].split(". ")[0]:
                    claim_parts[index] = f"The move is punished by {reply} in the analysed line: " + part[len(lead):]
                    merged = True
                    break
        if not merged:
            parts.append(f"The move is punished by {reply}.")
        parts += claim_parts
    else:
        parts += _render_claims(facts, variant)
    alt = _render_alternative(facts, variant)
    played = " ".join(p for p in parts if p).strip()
    if alt:
        # v19 comment_blocks: the alternative stands on its own line.
        played = f"{played}\n{alt.strip()}" if blocks else f"{played} {alt}".strip()
    return _explained_once(played) if _tidy() else played
