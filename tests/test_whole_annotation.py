import asyncio
import json
from types import SimpleNamespace

import chess
import pytest

from app.models.comment_facts import Claim, CommentFacts, EnvisionedLine
from app.core.commentary.phases.composer import compose_facts_comment, narrative


def facts():
    board = chess.Board()
    start = board.fen()
    board.push_san("e4")
    first = board.fen()
    board.push_san("e5")
    return CommentFacts(ply=1, san="e4", uci="e2e4", mover="White", phase="mid",
        verdict="holds the balance", eval_cp=15,
        display_line=EnvisionedLine(start_fen=start, line_uci=["e2e4", "e7e5"], line_san=["e4", "e5"],
            fens=[first, board.fen()], leaf_fen=board.fen()),
        claims=[Claim(rule_id="test", text="White has a pawn on e4.", baseline_fen=start,
                      support_fen=first, support_ply=1)])


def proposal():
    return {"text": "White advances the e-pawn; Black can answer with ...e5, keeping the balance.",
            "main": [0], "alternative": []}


class Service:
    _provider = SimpleNamespace(is_configured=lambda: True)

    def __init__(self, supported=True, raw=None):
        self.supported, self.raw, self.review_input = supported, raw, None

    async def _llm_call_json_schema(self, system, user, **kwargs):
        self.writer_input = json.loads(user)
        return self.raw or json.dumps(proposal())

    async def review_json(self, system, user, **kwargs):
        self.review_input = json.loads(user)
        return json.dumps({"sentences": [{"index": 0, "supported": self.supported, "reason": "Evidence checked."}]})


@pytest.mark.parametrize("supported", [True, False])
def test_default_whole_annotation_has_no_app_prose_and_requires_review(monkeypatch, supported):
    monkeypatch.setenv("COMMENTARY_LLM_REVIEW_ENABLED", "1")
    monkeypatch.delenv("COMMENTARY_RENDER_MODE", raising=False)
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    service = Service(supported)
    result = asyncio.run(compose_facts_comment(service, facts(), model=None, effort="low"))
    assert result["rendering_contract"] == narrative.VERSION
    assert result["rendering"] == ("llm" if supported else "template")
    assert result["contract_ok"] is supported
    if supported:
        assert result["text"] == proposal()["text"]
        assert "In the displayed line" not in result["text"]
        assert "Stockfish" not in result["text"]
    assert service.review_input["sentences"][0]["assessment"]["verdict"] == "holds the balance"


def test_writer_and_reviewer_share_entire_branch_not_only_claim_endpoint():
    f = facts()
    _, evidence = narrative.prompt(f, "intermediate")
    p = {"text": "White plays e4 and Black replies with the pawn from e7 to e5.", "main": [0], "alternative": []}
    _, review = narrative.parse_proposal(json.dumps(p), evidence)
    assert len(review[0]["traces"]["main"]) == 2
    assert review[0]["claims"]["main"][0]["support_ply"] == 1
    assert "claims" not in review[0]["board_context"]["main"]
    assert "pawn_files" not in review[0]["board_context"]["main"]["start"]


@pytest.mark.parametrize("change", [
    {"main": [99]}, {"main": [True]}, {"main": [[0]]}, {"main": [0, 0]},
    {"text": "White wins [%eval 9]."}, {"text": "White is +9.50."},
    {"text": "White is {{ALT_EVAL}}."}, {"text": "word " * (narrative.MAX_WORDS + 1)},
])
def test_invalid_citations_and_unprotected_values_rejected(change):
    _, evidence = narrative.prompt(facts(), "intermediate")
    p = dict(proposal(), **change)
    with pytest.raises(ValueError):
        narrative.parse_proposal(json.dumps(p), evidence)


def test_protected_numeric_value_is_resolved_before_review_and_render():
    f = facts()
    _, evidence = narrative.prompt(f, "intermediate")
    p = dict(proposal(), text="The position is balanced ({{MAIN_EVAL}}).")
    parsed, reviews = narrative.parse_proposal(json.dumps(p), evidence)
    assert reviews[0]["text"] == "The position is balanced (+0.15)."
    assert narrative.render(parsed, evidence) == reviews[0]["text"]


def test_previous_capture_and_development_are_verified_mechanical_relations():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    before = board.fen()
    board.push_san("exd5")
    start = board.fen()
    previous = {"fen_before": before, "fen_after": start, "uci": "e4d5", "san": "exd5"}
    fens = []
    for san in ("Qxd5", "Nc3"):
        board.push_san(san)
        fens.append(board.fen())
    f = CommentFacts(ply=4, san="Qxd5", uci="d8d5", mover="Black", phase="mid", verdict="holds the balance",
        display_line=EnvisionedLine(start_fen=start, line_uci=["d8d5", "b1c3"], line_san=["Qxd5", "Nc3"],
            fens=fens, leaf_fen=fens[-1]), selection_metadata={"previous_move": previous})
    _, evidence = narrative.prompt(f, "intermediate")
    assert {r["kind"] for r in evidence["mechanical_relations"]} == {"recapture", "development_move"}
    f.selection_metadata["previous_move"]["fen_after"] = chess.STARTING_FEN
    _, evidence = narrative.prompt(f, "intermediate")
    assert not any(r["kind"] == "recapture" for r in evidence["mechanical_relations"])


def test_empty_positional_catalog_can_explain_engine_assessment(monkeypatch):
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "grounded_rewrite")
    f = facts().model_copy(update={"claims": []})
    service = Service(raw=json.dumps({"text": "The position remains balanced.", "main": [], "alternative": []}))
    result = asyncio.run(compose_facts_comment(service, f, model=None, effort="low"))
    assert result["rendering"] == "llm"


def test_failed_draft_is_preserved_for_diagnosis(monkeypatch):
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "grounded_rewrite")
    service = Service(raw='{"text":"bad","main":[99],"alternative":[]}')
    result = asyncio.run(compose_facts_comment(service, facts(), model=None, effort="low"))
    assert result["raw_proposal"] == service.raw
    assert result["contract_ok"] is False
    assert service.review_input is None


@pytest.mark.parametrize("semantic_review", ["accepted", "disabled"])
def test_pipeline_preserves_screened_spacing_and_words(monkeypatch, semantic_review):
    from app.core.commentary.phases import composer
    from app.core.commentary.pipeline.move_pipeline import FactsComposeStage, MoveCommentaryContext

    text = "White keeps matters level. ...e5 is a comparable answer."
    async def compose(*args, **kwargs):
        return {"text": text, "rendering": "llm", "rendering_contract": narrative.VERSION,
                "semantic_review": semantic_review, "contract_ok": True}
    monkeypatch.setattr(composer, "compose_facts_comment", compose)
    event = SimpleNamespace(comment_facts=facts(), ply=1, san="e4", uci="e2e4",
                            fen_after=facts().display_line.fens[0], key_moment_type="critical_decision")
    context = MoveCommentaryContext(move_event=event, game_context=None, analyzed_row=None,
        service=SimpleNamespace(provider_name="cursor"), composer_effort="low", key_moment_type="critical_decision")
    asyncio.run(FactsComposeStage().run(context))
    assert context.final_text == text
    assert context.llm_debug["forbidden_phrase_hits"] == 1


@pytest.mark.parametrize("semantic_review", ["accepted", "disabled"])
def test_assembly_does_not_append_unreviewed_phase_sentence(semantic_review):
    from app.core.engine.analysis.comment_assembly import _resolve_move_comment

    text = "The exchange leaves an equal ending."
    analyzed = SimpleNamespace(hiddenFeatures={"_llm": {"comment": text,
        "rendering_contract": narrative.VERSION, "semantic_review": semantic_review}})
    event = SimpleNamespace(comment_facts=facts(), key_moment_type="critical_decision",
        phase_transition="queens_off", fen_after="7k/8/8/8/8/8/8/K7 w - - 0 1",
        eval_after_cp=0, eval_swing_cp=0, ply=1)
    resolved = _resolve_move_comment(SimpleNamespace(phase_raw="mid"), analyzed, event,
                                    "critical_decision", side_ok=True)
    assert resolved.comment == text


@pytest.mark.parametrize("mode", ["grounded_rewrite", "grounded_rewrite_v2"])
def test_review_disabled_by_default_and_validation_remains(monkeypatch, mode):
    from tests.test_grounded_rewrite import Service as LegacyService, facts as legacy_facts
    monkeypatch.delenv("COMMENTARY_LLM_REVIEW_ENABLED", raising=False)
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", mode)
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    service = Service(False) if mode == "grounded_rewrite" else LegacyService(False)
    f = facts() if mode == "grounded_rewrite" else legacy_facts()
    async def forbidden_review(*args, **kwargs):
        raise AssertionError("disabled reviewer was called")
    service.review_json = forbidden_review
    result = asyncio.run(compose_facts_comment(service, f, model=None, effort="low"))
    assert result["rendering"] == "llm"
    assert result["semantic_review"] == "disabled"
    assert result["commentary_status"] == "llm_unreviewed"
    assert result["review_scope"] == "deterministic_validation_only"
    async def invalid_writer(*args, **kwargs):
        return "not JSON"
    service._llm_call_json_schema = invalid_writer
    result = asyncio.run(compose_facts_comment(service, f, model=None, effort="low"))
    assert result["rendering"] == "template"
    assert not result["contract_ok"]
    assert result["commentary_status"] == "template_validation_failed"


def test_comment_status_pending_only_until_completion():
    from app.core.engine.analysis.comment_assembly import _comment_status
    move = SimpleNamespace(hiddenFeatures={})
    event = SimpleNamespace(comment_facts=None, brief_commentary=False, key_moment_type="mistake")
    for done, expected in [(False, "pending"), (True, "template")]:
        status = _comment_status(move, event, "floor", "mid", eligible=True, llm_done=done)
        assert status["has_comment"]
        assert status["commentary_status"] == expected
    event.comment_facts = SimpleNamespace(terminal_result="1-0")
    assert _comment_status(move, event, "mate", "mid", eligible=True,
                           llm_done=False)["commentary_status"] == "template"


@pytest.mark.parametrize("phase,comment", [("early", "English Opening"),
                                          ("mid", "White develops the knight."),
                                          ("mid", None)])
def test_legacy_saved_move_does_not_invent_generation_provenance(phase, comment):
    from app.models.GameJson import GameMove
    payload = dict(mn=1, color="w", san="c4", uci="c2c4", fen=chess.STARTING_FEN,
                   phase=phase, comment=comment)
    saved = GameMove.model_validate(payload).model_dump()
    assert saved["has_comment"] is bool(comment)
    assert saved["comment_kind"] is None
    assert saved["commentary_status"] is None
    assert saved["commentary_reason"] is None
    explicit = GameMove.model_validate(dict(payload, has_comment=False,
        comment_kind="template", commentary_status="template_validation_failed"))
    assert explicit.has_comment is False
    assert explicit.commentary_status == "template_validation_failed"


def test_writer_sees_placeholder_names_not_numeric_scores():
    user, evidence = narrative.prompt(facts(), "beginner")
    shown = json.loads(user)["evidence"]
    assert "eval_cp_white" not in shown["assessment"] and "substitutions" not in shown
    assert shown["placeholders"] == ["MAIN_EVAL"]
    assert shown["citable_ids"] == {"main": [0], "alternative": []}
    assert evidence["assessment"]["eval_cp_white"] == 15 and evidence["substitutions"] == {"MAIN_EVAL": "+0.15"}
    text = narrative.render(*narrative.parse_proposal(json.dumps(
        {"text": "White keeps the balance at {{MAIN_EVAL}}.", "main": [0], "alternative": []}), evidence)[:1], evidence)
    assert text == "White keeps the balance at +0.15."


def test_validator_allows_necessity_wording_and_long_text_up_to_cap():
    _, evidence = narrative.prompt(facts(), "beginner")
    def check(text):
        return narrative.parse_proposal(json.dumps({"text": text, "main": [0], "alternative": []}), evidence)
    check("White must keep the e-pawn, and Black has to answer; it is the only move.")
    check(" ".join(["word"] * narrative.MAX_WORDS))
    with pytest.raises(ValueError, match="word budget"):
        check(" ".join(["word"] * (narrative.MAX_WORDS + 1)))
