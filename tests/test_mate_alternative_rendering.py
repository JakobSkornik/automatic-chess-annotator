import chess
import pytest

from app.models.comment_facts import BestAlternative, CommentFacts, Claim
from app.core.commentary.phases.composer.framing import alternative_comparison, alt_gap_cp
from app.core.commentary.phases.composer.template import render_facts_template
from app.core.commentary.rules.material import describe_material


def facts(**kwargs):
    values = dict(ply=71, san="Qe6", uci="e5e6", mover="White", phase="end",
                  verdict="allows forced mate", eval_mate=-5,
                  better_alternative=BestAlternative(san="Rxe5", uci="e1e5", eval_cp=-741))
    values.update(kwargs)
    return CommentFacts(**values)


@pytest.mark.parametrize("mover,played,alternative,expected", [
    ("White", -5, None, "better"), ("Black", -5, None, "weaker"),
    ("White", 5, None, "weaker"), ("Black", 5, None, "better"),
    ("White", -5, -8, "better"), ("White", 5, 8, "weaker"),
])
def test_mate_ordering(mover, played, alternative, expected):
    f = facts(mover=mover, eval_mate=played,
              better_alternative=BestAlternative(san="Rxe5", uci="e1e5", eval_cp=-741,
                                                 eval_mate=alternative))
    assert alternative_comparison(f) == expected
    assert alt_gap_cp(f) is None


def test_mate_to_finite_alternative_is_better_without_numeric_gap():
    text = render_facts_template(facts())
    assert "better move was Rxe5" in text
    assert "comparable" not in text
    assert "pawns better" not in text


def test_missing_scores_are_neutral():
    f = facts(eval_mate=None, eval_cp=None)
    assert alternative_comparison(f) == "unknown"
    assert "Another alternative was Rxe5" in render_facts_template(f)
    assert "comparable" not in render_facts_template(f)


def test_material_surplus_is_explicit():
    board = chess.Board("7k/8/8/8/8/8/pb6/7K w - - 0 1")
    assert describe_material(board, changed=False)[0] == "Black is a bishop and a pawn up."


def test_exchange_claim_avoids_double_prefix():
    f = facts(better_alternative=None, claims=[Claim(rule_id="exchange_sequence",
        text="In the displayed exchange on e5, White loses a rook, while Black loses a bishop.",
        realization="envisioned")])
    text = render_facts_template(f)
    assert "continuation, In" not in text
    assert "In the displayed exchange on e5" in text


def test_engine_choice_uses_concise_verdict():
    assert "engine's preferred move" not in render_facts_template(facts(), archetype="engine_choice")
