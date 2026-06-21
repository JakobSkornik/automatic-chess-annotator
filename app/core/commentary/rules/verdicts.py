"""Eval (transition) -> human verdict phrasing."""

from __future__ import annotations

# Advantage-bracket boundaries (|centipawns|): equality < EQUAL < slight < SLIGHT
# < clear < CLEAR < winning < WINNING <= decisive.
EQUAL_MAX_CP = 25
SLIGHT_MAX_CP = 75
CLEAR_MAX_CP = 150
WINNING_MAX_CP = 300
TRANSITION_MIN_CP = 35  # eval change below this = "nothing really changed"


def _adv_bracket(cp: int) -> str | None:
    """Advantage label for |cp| >= EQUAL_MAX_CP, else None (equality)."""
    mag = abs(cp)
    if mag < EQUAL_MAX_CP:
        return None
    if mag < SLIGHT_MAX_CP:
        return "a slight advantage"
    if mag < CLEAR_MAX_CP:
        return "a clear advantage"
    if mag < WINNING_MAX_CP:
        return "a winning advantage"
    return "a decisive advantage"


def verdict_for_transition(
    before_cp: int | None,
    after_cp: int | None,
    eval_mate: int | None,
    mover: str,
) -> str:
    """Verdict phrased as the eval *story* of the move from the mover's seat:
    held / extended / slipped / thrown away / conceded — far more informative
    than the absolute bracket alone. Falls back to ``verdict_for_eval``."""
    if eval_mate is not None:
        side = "White" if eval_mate > 0 else "Black"
        return f"leads to forced mate for {side}"
    if before_cp is None or after_cp is None:
        return verdict_for_eval(after_cp, eval_mate)

    opp = "Black" if mover == "White" else "White"
    sign = 1 if mover == "White" else -1
    b = sign * int(before_cp)  # mover POV
    a = sign * int(after_cp)
    d = a - b
    a_lab = _adv_bracket(a)
    _adv_bracket(b)

    if abs(d) < TRANSITION_MIN_CP:  # nothing really changed
        if a_lab is None:
            return "holds the balance"
        if a > 0:
            return f"maintains {mover}'s advantage"
        return verdict_for_eval(after_cp)

    if d < 0:  # the mover lost ground
        if b > EQUAL_MAX_CP and a > EQUAL_MAX_CP:
            return f"lets {mover}'s advantage shrink"
        if b > EQUAL_MAX_CP and -EQUAL_MAX_CP <= a <= EQUAL_MAX_CP:
            return "lets the advantage slip away to equality"
        if b > EQUAL_MAX_CP and a < -EQUAL_MAX_CP:
            return f"throws away the advantage and hands {opp} {_adv_bracket(-a)}"
        if -EQUAL_MAX_CP <= b <= EQUAL_MAX_CP and a < -EQUAL_MAX_CP:
            return f"concedes {_adv_bracket(-a)} to {opp}"
        if b < -EQUAL_MAX_CP and a < b:
            return f"makes matters worse — {opp} now has {_adv_bracket(-a)}"
        return verdict_for_eval(after_cp)

    # the mover gained ground (usually cashing in on the opponent's error)
    if a > EQUAL_MAX_CP and b <= EQUAL_MAX_CP:
        return f"seizes {a_lab} for {mover}"
    if a > EQUAL_MAX_CP and b > EQUAL_MAX_CP:
        return f"extends {mover}'s advantage"
    if -EQUAL_MAX_CP <= a <= EQUAL_MAX_CP:
        return "restores the balance"
    return f"fights back, though {opp} keeps {_adv_bracket(-a)}"


def verdict_for_eval(eval_cp: int | None, eval_mate: int | None = None) -> str:
    if eval_mate is not None:
        side = "White" if eval_mate > 0 else "Black"
        return f"leads to forced mate for {side}"
    if eval_cp is None:
        return ""
    cp = int(eval_cp)
    mag = abs(cp)
    side = "White" if cp > 0 else "Black"
    if mag < EQUAL_MAX_CP:
        return "leads to equality"
    if mag < SLIGHT_MAX_CP:
        return f"leads to a slight advantage for {side}"
    if mag < CLEAR_MAX_CP:
        return f"leads to a clear advantage for {side}"
    if mag < WINNING_MAX_CP:
        return f"gives {side} a winning advantage"
    return f"leaves {side} with a decisive advantage"


# ---------------------------------------------------------------------------
# CommentFacts builder
# ---------------------------------------------------------------------------
