"""Shared fact tokens: the verbatim eval/PV strings and move labels that every
rendering path (template and LLM) copies unchanged."""

from __future__ import annotations

from app.models.comment_facts import BestAlternative, CommentFacts

EVAL_STORY_MIN_CP = 50  # show before->after eval story when it moved this much


def eval_token(facts: CommentFacts) -> str:
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
    move_no = (facts.ply + 1) // 2
    dots = "." if facts.mover == "White" else "..."
    return f"{move_no}{dots}{facts.san}"


def _gerundize(verdict: str) -> str:
    """'leads to equality' -> 'leading to equality' (for 'A better move was X, ...')."""
    for head, ger in (("leads", "leading"), ("gives", "giving"), ("leaves", "leaving")):
        if verdict.startswith(head + " "):
            return ger + verdict[len(head) :]
    return verdict
