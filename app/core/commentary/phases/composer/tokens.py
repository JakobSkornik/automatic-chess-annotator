"""Shared fact tokens: the verbatim eval/PV strings and move labels that every
rendering path (template and LLM) copies unchanged."""

from __future__ import annotations

import re

from app.models.comment_facts import BestAlternative, CommentFacts

EVAL_STORY_MIN_CP = 50  # show before->after eval story when it moved this much


def eval_token(facts: CommentFacts) -> str:
    if facts.terminal_result is not None:
        return f"(Result: {facts.terminal_result})"
    if facts.eval_mate is not None:
        side = "White" if facts.eval_mate > 0 else "Black"
        return f"(#{abs(facts.eval_mate)} for {side}, {facts.engine}:{facts.depth})"
    if facts.eval_cp is None:
        return ""
    # Show the eval STORY when the move moved the needle.
    if (
        facts.eval_before_cp is not None
        and abs(facts.eval_cp - facts.eval_before_cp) >= EVAL_STORY_MIN_CP
    ):
        return (
            f"({facts.eval_before_cp / 100:+.2f} → {facts.eval_cp / 100:+.2f}, "
            f"{facts.engine}:{facts.depth})"
        )
    return f"({facts.eval_cp / 100:+.2f}, {facts.engine}:{facts.depth})"


def pv_token(facts: CommentFacts) -> str:
    line = facts.display_line
    if not line or not line.line_san:
        return ""
    return "[pv:" + " ".join(line.line_san) + "]"


def _alt_pv_token(alt: BestAlternative) -> str:
    if not alt.display_line or not alt.display_line.line_san:
        return ""
    return "[pv:" + " ".join(alt.display_line.line_san) + "]"


def _move_label(facts: CommentFacts) -> str:
    move_no = facts.move_number or (facts.ply + 1) // 2
    dots = "." if facts.mover == "White" else "..."
    return f"{move_no}{dots}{facts.san}"


def alternative_label(facts: CommentFacts) -> str:
    """The alternative with its move number (it is a move of the same side as the played one)."""
    return _move_label(facts)[: -len(facts.san)] + facts.better_alternative.san


def reply_label(facts: CommentFacts, san: str) -> str:
    """The opponent's reply to the played move, with its move number."""
    move_no = facts.move_number or (facts.ply + 1) // 2
    return f"{move_no}...{san}" if facts.mover == "White" else f"{move_no + 1}.{san}"


_KEEP_CAPITAL = re.compile(r"(White|Black|I)\b|O-O|[KQRBN][a-h1-8x]")


def lower_start(text: str) -> str:
    """A statement that follows a lead-in ('After 27...Rxh2 in the analysed line, ...')
    starts in lower case, unless it starts with a side, a piece move or a castling."""
    if not text or _KEEP_CAPITAL.match(text):
        return text
    return text[0].lower() + text[1:]


def _gerundize(verdict: str) -> str:
    """'leads to equality' -> 'leading to equality' (for 'A better move was X, ...')."""
    for head, ger in (("leads", "leading"), ("gives", "giving"), ("leaves", "leaving")):
        if verdict.startswith(head + " "):
            return ger + verdict[len(head) :]
    return verdict
