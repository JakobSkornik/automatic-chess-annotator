import json

import chess
import pytest

from app.models.comment_facts import Claim, CommentFacts, EnvisionedLine
from app.core.commentary.phases.composer.sentence_plan import catalog, default_plan, render_plan, has_editorial_choice


def facts():
    board = chess.Board()
    start = board.fen()
    board.push_san("e4")
    support = board.fen()
    line = EnvisionedLine(start_fen=start, line_uci=["e2e4"], line_san=["e4"], fens=[support], leaf_fen=support)
    claim = Claim(rule_id="example", text="White advances the e-pawn.", beneficiary="white",
                  support_ply=1, support_fen=support, baseline_fen=start)
    return CommentFacts(ply=1, san="e4", uci="e2e4", mover="White", phase="mid",
                        verdict="holds the balance", display_line=line, claims=[claim])


def test_valid_realization_keeps_assertion_and_anchor():
    f = facts()
    result = render_plan(json.dumps(default_plan(catalog(f))), f)
    assert "After the played move, White advances the e-pawn." in result["text"]
    assert result["sentence_map"][0]["support_fen"] == f.claims[0].support_fen
    assert "not_independent_rule_truth" in result["verification_scope"]
    assert not has_editorial_choice(catalog(f))


@pytest.mark.parametrize("plan", [
    {"main": [], "alternative": []},
    {"main": [{"ids": [0, 0], "form": "change"}], "alternative": []},
    {"main": [{"ids": [True], "form": "change"}], "alternative": []},
    {"main": [{"ids": [9], "form": "change"}], "alternative": []},
    {"main": [{"ids": [0], "form": "change", "text": "Black has an extra queen."}], "alternative": []},
])
def test_missing_duplicate_wrong_or_invented_assertions_rejected(plan):
    with pytest.raises(ValueError):
        render_plan(json.dumps(plan), facts())


@pytest.mark.parametrize("mutation", ["side", "support", "line", "timing"])
def test_wrong_side_position_line_and_time_rejected(mutation):
    f = facts()
    if mutation == "side": f.mover = "Black"
    if mutation == "support": f.claims[0].support_fen = chess.STARTING_FEN
    if mutation == "line": f.display_line.line_san = ["d4"]
    if mutation == "timing": f.claims[0].realization = "envisioned"
    with pytest.raises(ValueError):
        catalog(f)


def test_cannot_merge_opposite_concession_roles():
    f = facts()
    f.claims.append(f.claims[0].model_copy(update={"is_concession": True}))
    with pytest.raises(ValueError, match="concession"):
        render_plan(json.dumps({"main": [{"ids": [0, 1], "form": "change"}], "alternative": []}), f)


def test_valid_merge_keeps_both_source_claims():
    f = facts()
    f.claims.append(f.claims[0].model_copy(update={"text": "The pawn now occupies e4."}))
    result = render_plan(json.dumps({"main": [{"ids": [1, 0], "form": "change"}], "alternative": []}), f)
    assert "The pawn now occupies e4; White advances the e-pawn." in result["text"]
    assert result["sentence_map"][0]["claim_ids"] == [1, 0]
