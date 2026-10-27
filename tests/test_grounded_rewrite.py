"""Grounded prose acceptance, rejection, branch timing and session isolation."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chess
import pytest

from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.phases.composer import compose_facts_comment
from app.core.commentary.phases.composer import rewrite
from app.core.commentary.phases.composer.grounded import render_editor_response
from app.core.commentary.phases.composer.template import render_facts_template
from app.models.comment_facts import BestAlternative, Claim, CommentFacts, EnvisionedLine


def facts():
    board = chess.Board()
    start = board.fen()
    board.push_san("e4")
    first = board.fen()
    board.push_san("e5")
    second = board.fen()
    return CommentFacts(
        ply=1, san="e4", uci="e2e4", mover="White", phase="mid", verdict="holds the balance",
        eval_cp=0, display_line=EnvisionedLine(
            start_fen=start, line_uci=["e2e4", "e7e5"], line_san=["e4", "e5"],
            fens=[first, second], leaf_fen=second),
        claims=[
            Claim(rule_id="test", text="White advances the pawn to e4.",
                  text_state="White has a pawn on e4.", beneficiary="white",
                  baseline_fen=start, support_fen=first, support_ply=1),
            Claim(rule_id="test", text="Black advances the pawn to e5.",
                  text_state="Black has a pawn on e5.", beneficiary="black", is_concession=True,
                  baseline_fen=start, support_fen=second, support_ply=2, realization="envisioned"),
        ])


def proposal():
    return {"main": [{"ids": [0], "text": "White's pawn advances to e4."},
                     {"ids": [1], "text": "Black's pawn stands on e5."}], "alternative": []}


def review(supported=True):
    return {"sentences": [{"index": i, "supported": supported, "reason": "Matches source."}
                          for i in range(2)]}


class Service:
    _provider = SimpleNamespace(is_configured=lambda: True)

    def __init__(self, accepted=True):
        self.accepted = accepted

    async def _llm_call_json_schema(self, *args, **kwargs):
        return json.dumps(proposal())

    async def review_json(self, *args, **kwargs):
        return json.dumps(review(self.accepted))


@pytest.mark.parametrize("accepted", [True, False])
def test_legacy_rewrite_accepts_only_after_semantic_review(monkeypatch, accepted):
    monkeypatch.setenv("COMMENTARY_LLM_REVIEW_ENABLED", "1")
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "grounded_rewrite_v2")
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    f = facts()
    result = asyncio.run(compose_facts_comment(Service(accepted), f, model=None, effort="low"))
    assert result["rendering_contract"] == rewrite.VERSION
    assert result["contract_ok"] is accepted
    if accepted:
        assert result["rendering"] == "llm"
        assert "In the displayed line, Black's pawn stands on e5." in result["text"]
    else:
        assert result["text"] == render_facts_template(f, archetype="neutral")
        assert result["semantic_review"] == "rejected_or_unavailable"


@pytest.mark.parametrize("kind", ["square", "id", "duplicate", "token"])
def test_proposal_cannot_cross_evidence_boundaries(kind):
    _, evidence = rewrite.prompt(facts(), "intermediate")
    p = proposal()
    if kind == "square":
        p["main"][0]["text"] = "White wins on h8."
    elif kind == "id":
        p["main"][0]["ids"] = [99]
    elif kind == "duplicate":
        p["main"][1]["ids"] = [0]
    else:
        p["main"][0]["text"] = "White wins [%eval 9]."
    with pytest.raises(ValueError):
        rewrite.parse_proposal(json.dumps(p), evidence)


def test_semantic_review_must_cover_every_sentence():
    with pytest.raises(ValueError):
        rewrite.validate_review(json.dumps({"sentences": review()["sentences"][:1]}), 2)
    invalid = review()
    invalid["sentences"][1]["index"] = 0
    with pytest.raises(ValueError):
        rewrite.validate_review(json.dumps(invalid), 2)


def test_selection_can_omit_redundant_merits():
    f = facts()
    f.claims.append(f.claims[0].model_copy())
    _, evidence = rewrite.prompt(f, "expert")
    parsed, sentences = rewrite.parse_proposal(json.dumps(proposal()), evidence)
    assert len(sentences) == 2
    assert parsed == proposal()


def test_cross_anchor_explanation_supplies_chronology_and_rendered_scope():
    f = facts()
    _, evidence = rewrite.prompt(f, "intermediate")
    p = {"main": [{"ids": [0, 1], "text": "White advances to e4, followed by Black's pawn advance to e5."}], "alternative": []}
    parsed, reviews = rewrite.parse_proposal(json.dumps(p), evidence)
    assert [e["anchor"] for e in reviews[0]["legal_trace"]] == ["1.e4", "1...e5"]
    assert reviews[0]["rendering_prefix"] == "In the displayed line, "
    assert reviews[0]["branch_role"] == "played_move"
    assert "In the displayed line, White advances" in rewrite.render(parsed, evidence, f, "neutral")


def test_trace_allows_only_mechanical_context_through_latest_cited_anchor():
    _, evidence = rewrite.prompt(facts(), "intermediate")
    p = {"main": [{"ids": [0], "text": "White moves the pawn from e2 to e4."}], "alternative": []}
    _, reviews = rewrite.parse_proposal(json.dumps(p), evidence)
    assert len(reviews[0]["legal_trace"]) == 1
    assert reviews[0]["legal_trace"][0]["gives_check"] is False
    p["main"][0]["text"] = "Black's pawn moves from e7 to e5."
    with pytest.raises(ValueError, match="Uncited square"):
        rewrite.parse_proposal(json.dumps(p), evidence)


def test_knight_mobility_implication_requires_absolute_pin_on_support_board():
    board = chess.Board("7k/8/8/8/1b6/2N5/P7/4K3 w - - 0 1")
    start = board.fen()
    board.push_san("a3")
    f = facts().model_copy(update={"san": "a3", "uci": "a2a3",
        "display_line": EnvisionedLine(start_fen=start, line_uci=["a2a3"], line_san=["a3"],
            fens=[board.fen()], leaf_fen=board.fen()),
        "claims": [Claim(rule_id="absolute_pin", text="Black's bishop on b4 pins White's knight on c3 to the king on e1.",
            baseline_fen=start, support_fen=board.fen(), support_ply=1)]})
    _, evidence = rewrite.prompt(f, "intermediate")
    assert evidence["main"][0]["implications"] == ["White's knight on c3 cannot legally move while this absolute pin remains."]
    f.claims[0].text = "Black's bishop on b4 has a pin."
    _, evidence = rewrite.prompt(f, "intermediate")
    assert not evidence["main"][0]["implications"]


def test_concrete_opposing_reply_must_survive_selection():
    f = facts()
    f.claims[1].rule_id = "legal_capture"
    _, evidence = rewrite.prompt(f, "intermediate")
    p = proposal()
    p["main"] = p["main"][:1]
    with pytest.raises(ValueError, match="concrete"):
        rewrite.parse_proposal(json.dumps(p), evidence)


def test_incidental_concession_can_be_omitted():
    _, evidence = rewrite.prompt(facts(), "intermediate")
    p = proposal()
    p["main"] = p["main"][:1]
    rewrite.parse_proposal(json.dumps(p), evidence)


def test_hanging_piece_is_not_a_mandatory_played_reply():
    f = facts()
    f.claims[0].rule_id = "legal_capture"
    f.claims[1].rule_id = "hanging_piece"
    _, evidence = rewrite.prompt(f, "intermediate")
    p = proposal()
    p["main"] = p["main"][:1]
    rewrite.parse_proposal(json.dumps(p), evidence)


def test_late_single_anchor_does_not_relocate_earlier_trace_events():
    _, evidence = rewrite.prompt(facts(), "intermediate")
    p = {"main": [{"ids": [1], "text": "White first moves e4, then Black plays e5."}], "alternative": []}
    _, reviews = rewrite.parse_proposal(json.dumps(p), evidence)
    assert reviews[0]["rendering_prefix"] == "In the displayed line, "
    assert len(reviews[0]["legal_trace"]) == 2


def test_writer_receives_evidence_beyond_legacy_audience_cap(monkeypatch):
    monkeypatch.setenv("COMMENTARY_LLM_REVIEW_ENABLED", "1")
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "grounded_rewrite_v2")
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    f = facts()
    f.claims.extend([f.claims[0].model_copy(), f.claims[0].model_copy(update={"rule_id": "absolute_pin"})])
    class InspectService(Service):
        async def _llm_call_json_schema(self, system, user, **kwargs):
            assert len(json.loads(user)["catalog"]["main"]) == 4
            return await super()._llm_call_json_schema(system, user, **kwargs)
    result = asyncio.run(compose_facts_comment(InspectService(), f, model=None, effort="low"))
    assert len(result["claim_catalog"]["main"]) == 4


def test_bounded_editor_respects_selected_forecast_change_form():
    f = facts()
    raw = {"main": [{"id": 0, "form": "state"}, {"id": 1, "form": "change"}], "alternative": []}
    text = render_editor_response(json.dumps(raw), f, "expert", "neutral")
    assert "White has a pawn on e4." in text
    assert "Black advances the pawn to e5." in text


@pytest.mark.parametrize("cp, missed", [(49, False), (50, True), (None, False)])
def test_missed_improvement_does_not_require_alternative_claims(cp, missed):
    f = facts()
    f.better_alternative = BestAlternative(san="d4", uci="d2d4", eval_cp=cp)
    text = render_facts_template(f)
    assert ("missed a better continuation" in text) is missed


def test_reviewer_uses_independent_calls_without_advancing_writer_session(monkeypatch):
    monkeypatch.setenv("COMMENTARY_LLM_REVIEW_ENABLED", "1")
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "grounded_rewrite_v2")
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    count = 0

    def respond(**kwargs):
        nonlocal count
        count += 1
        is_review = kwargs["text"]["format"]["name"] == "grounded_rewrite_review"
        return SimpleNamespace(id=f"response-{count}", output_text=json.dumps(
            review() if is_review else proposal()), usage=None)

    provider = SimpleNamespace(name="openai", is_configured=lambda: True,
        _client=SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=respond))))
    # Independent review calls use the standard provider adapter.
    from app.core.commentary.llm_providers import OpenAIProvider
    backend = OpenAIProvider.__new__(OpenAIProvider)
    backend._client = provider._client
    monkeypatch.setattr(backend, "is_configured", lambda: True)
    service = AdvancedCommentService(provider=backend)

    async def run():
        async with service.commentary_session("rewrite-test") as session:
            for ply in (2, 1):
                result = await compose_facts_comment(session, facts(), model="gpt-4.1", effort="low")
                assert result["rendering"] == "llm", result
                await session.record_annotation(dict(result, ply=ply))
    asyncio.run(run())
    calls = provider._client.responses.create.call_args_list
    assert len(calls) == 4
    assert "previous_response_id" not in calls[1].kwargs
    assert calls[2].kwargs["previous_response_id"] == "response-1"
    assert "previous_response_id" not in calls[3].kwargs


def test_review_provenance_survives_game_debug_serialization():
    from app.core.commentary.pipeline.game_pipeline import _write_llm_comment
    from app.core.engine.analysis.comment_assembly import _build_move_debug
    from app.models.chess_events import MoveEvent

    f = facts()
    analyzed = SimpleNamespace(hiddenFeatures={})
    event = MoveEvent(move_index=0, ply=1, san=f.san, uci=f.uci,
                      fen_before=f.display_line.start_fen, fen_after=f.display_line.fens[0],
                      phase="mid", move_quality="good", event_type="quiet")
    debug = {"facts_contract_ok": False, "facts_renderings": {"intermediate": "template"},
             "rendering_contract": rewrite.VERSION, "semantic_review": "rejected_or_unavailable",
             "fallback_reason": "Unsupported sentence", "selected_claims": proposal()}
    _write_llm_comment([SimpleNamespace(ply=1, analyzed_move=analyzed)], 1, "fallback", debug)
    saved = _build_move_debug(analyzed, event, f, "fallback", "mid")
    assert saved["rendering_contract"] == rewrite.VERSION
    assert saved["commentary_validation"]["fallback_reason"] == "Unsupported sentence"
    assert saved["commentary_validation"]["selected_claims"] == proposal()


def test_template_keeps_reply_before_follow_through():
    f = facts()
    f.claims = [f.claims[0], f.claims[1].model_copy(update={
        "text": "White follows through.", "text_state": None, "support_ply": 3,
        "beneficiary": "white", "is_concession": False}), f.claims[1]]
    text = render_facts_template(f)
    assert text.index("Black has a pawn on e5.") < text.index("White follows through.")
