"""Integration checks for verified context without relaxing the output contract."""
import asyncio
import json
from types import SimpleNamespace

import chess
import pytest

from app.core.commentary.phases.composer import compose_facts_comment
from app.core.commentary.phases.composer.grounded import (
    EDITOR_VERSION, editor_prompt, render_editor_response,
)
from app.core.commentary.phases.composer.sentence_plan import (
    PROMPT_VERSION, catalog, planner_prompt,
)
from app.models.comment_facts import Claim, CommentFacts, EnvisionedLine


@pytest.fixture(autouse=True)
def bounded_comparison_mode(monkeypatch):
    """This suite exercises the retained bounded editor's context ablation."""
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "bounded")


def facts():
    board = chess.Board()
    start = board.fen()
    board.push_san("e4")
    end = board.fen()
    return CommentFacts(
        ply=1, san="e4", uci="e2e4", mover="White", phase="mid",
        verdict="holds the balance", eval_cp=10,
        display_line=EnvisionedLine(
            start_fen=start, line_uci=["e2e4"], line_san=["e4"],
            fens=[end], leaf_fen=end,
        ),
        claims=[Claim(
            rule_id="example", text="White advances the e-pawn.",
            text_state="White's e-pawn stands on e4.", beneficiary="white",
            support_ply=1, support_fen=end, baseline_fen=start,
        )],
    )


def test_ablation_preserves_selections_but_adds_replayed_board_context():
    f = facts()
    basic = json.loads(editor_prompt(f, "intermediate", context_profile="claims_only"))
    enriched = json.loads(editor_prompt(f, "intermediate"))
    assert "evidence" not in basic
    assert {key: enriched[key] for key in basic} == basic
    assert enriched["prompt_version"] == EDITOR_VERSION
    node = enriched["evidence"]["branches"]["main"]
    assert node["claims"][0]["claim_index"] == basic["main"][0]["id"]
    assert "e4" in node["board_after_move"]["pieces"]["white"]["pawn"]
    assert node["claims"][0]["node_anchor"] == "1.e4"


@pytest.mark.parametrize("mutation", ["line", "support", "timing"])
def test_conflicting_evidence_rejected_before_prompting(mutation):
    f = facts()
    if mutation == "line":
        f.display_line.line_uci = ["e2e5"]
    elif mutation == "support":
        f.claims[0].support_fen = chess.STARTING_FEN
    else:
        f.claims[0].realization = "envisioned"
    with pytest.raises(ValueError):
        editor_prompt(f, "intermediate")


def test_legacy_missing_evidence_remains_explicitly_unavailable():
    f = facts()
    f.claims[0].support_fen = None
    branch = json.loads(editor_prompt(f, "intermediate"))["evidence"]["branches"]["main"]
    assert branch["claims"] == []
    assert branch["unavailable_claims"][0]["reason"] == "missing_support_fen"


def test_sentence_plan_receives_same_catalog_ids_with_board_evidence():
    f = facts()
    base = catalog(f)
    payload = planner_prompt(f, "expert")
    assert {key: payload[key] for key in base} == base
    assert payload["prompt_version"] == PROMPT_VERSION
    assert payload["board_context"]["branches"]["main"]["board_after_move"]["fen"] == f.display_line.leaf_fen


@pytest.mark.parametrize("response", [
    {"main": [{"id": 0, "form": "change", "text": "White wins a rook."}], "alternative": []},
    {"main": [], "alternative": [{"id": 0, "form": "change"}]},
    {"main": [{"id": 7, "form": "change"}], "alternative": []},
])
def test_richer_context_cannot_authorize_new_assertions(response):
    with pytest.raises(ValueError):
        render_editor_response(json.dumps(response), facts(), "expert", "neutral")


class Service:
    _provider = SimpleNamespace(is_configured=lambda: True)

    def __init__(self):
        self.calls = []

    async def _llm_call_json_schema(self, system, user, **kwargs):
        self.calls.append(json.loads(user))
        return json.dumps({"main": [{"id": 0, "form": "state"}], "alternative": []})


def test_production_uses_versioned_context_and_records_prompt_version(monkeypatch):
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    monkeypatch.delenv("COMMENTARY_CONTEXT_PROFILE", raising=False)
    service = Service()
    result = asyncio.run(compose_facts_comment(service, facts(), model=None, effort="low"))
    assert result["rendering"] == "llm"
    assert result["prompt_version"] == EDITOR_VERSION
    assert service.calls[0]["context_profile"] == "board_grounded"


def test_no_choice_skips_provider_without_losing_beginner_explanation(monkeypatch):
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    f = facts()
    f.claims[0].text_state = None
    f.claims[0].rule_id = "intent_opens_file"
    service = Service()
    result = asyncio.run(compose_facts_comment(service, f, model=None, effort="low", level="beginner"))
    expected = render_editor_response(
        json.dumps({"main": [{"id": 0, "form": "change"}], "alternative": []}),
        f, "beginner", "neutral",
    )
    assert service.calls == []
    assert result["rendering"] == "template"
    assert result["text"] == expected


def test_production_rejects_conflicting_context_without_call(monkeypatch):
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    monkeypatch.delenv("COMMENTARY_CONTEXT_PROFILE", raising=False)
    f = facts()
    f.claims[0].support_fen = chess.STARTING_FEN
    service = Service()
    result = asyncio.run(compose_facts_comment(service, f, model=None, effort="low"))
    assert service.calls == []
    assert result["rendering"] == "template"
    assert result["contract_ok"] is False
