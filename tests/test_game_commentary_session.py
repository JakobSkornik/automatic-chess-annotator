"""Game-level session lifecycle, reverse-order continuity and failure isolation."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chess
import pytest

from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.pipeline import game_pipeline as pipeline
from app.models.GameJson import GameMetadata
from app.models.chess_events import GameAnalysisContext, MoveEvent, MoveEventType, MoveQuality
from app.models.comment_facts import Claim, CommentFacts, EnvisionedLine


def state(game_id):
    board = chess.Board()
    events = []
    for ply, san in enumerate(("e4", "e5"), 1):
        before = board.fen()
        mover = "White" if board.turn else "Black"
        uci = board.parse_san(san).uci()
        board.push_san(san)
        after = board.fen()
        facts = CommentFacts(
            ply=ply, san=san, uci=uci, mover=mover, phase="early",
            verdict="holds the balance", eval_cp=0,
            display_line=EnvisionedLine(start_fen=before, line_uci=[uci], line_san=[san],
                                       fens=[after], leaf_fen=after),
            claims=[Claim(rule_id="example", text=f"{mover} advances the e-pawn.",
                          text_state=f"{mover}'s e-pawn stands on {san}.",
                          beneficiary=mover.lower(), support_ply=1,
                          support_fen=after, baseline_fen=before)],
        )
        events.append(MoveEvent(
            move_index=ply - 1, ply=ply, san=san, uci=uci,
            fen_before=before, fen_after=after, phase="opening",
            move_quality=MoveQuality.GOOD, event_type=MoveEventType.QUIET,
            key_moment_type="critical_decision", comment_facts=facts,
        ))
    return SimpleNamespace(
        metadata=GameMetadata(id=game_id, white="White", black="Black", result="*"),
        move_events=events, analyzed_rows=[],
        context=GameAnalysisContext(move_events=events), llm_done=False,
    )


def provider(responses=None):
    count = 0

    def reply(**kwargs):
        nonlocal count
        count += 1
        text = (responses[count - 1] if responses else
                json.dumps({"main": [{"id": 0, "form": "state"}], "alternative": []}))
        return SimpleNamespace(id=f"response-{count}", output_text=text, usage=None)

    return SimpleNamespace(
        name="openai", is_configured=lambda: True,
        _client=SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=reply))),
    )


def configure(monkeypatch):
    monkeypatch.setenv("HUMANIZATION_LEVEL", "2")
    monkeypatch.setenv("COMMENTARY_RENDER_MODE", "bounded")
    monkeypatch.delenv("COMMENTARY_CONTEXT_PROFILE", raising=False)
    monkeypatch.setattr(pipeline, "_promote_turning_points", lambda *args: None)
    monkeypatch.setattr(pipeline, "_apply_back_to_back_key_moment_suppression", lambda *args: None)
    monkeypatch.setattr(pipeline, "_pv_line_for_ai_payload", lambda row: [])


def test_game_pass_uses_reverse_order_and_keeps_sessions_isolated(monkeypatch):
    configure(monkeypatch)
    backend = provider()
    service = AdvancedCommentService(provider=backend)
    updates = []

    async def callback(event, data):
        if event == "AI_COMMENT_UPDATE":
            updates.append(data)

    async def run():
        for game_id in ("game-A", "game-B"):
            game = state(game_id)
            await pipeline.GameAnnotationPipeline().run_llm_phases(
                game, service, commentary_callback=callback,
            )
            assert game.llm_done

    asyncio.run(run())
    calls = backend._client.responses.create.call_args_list
    assert len(calls) == 4
    assert [u["moveId"] for u in updates] == [2, 1, 2, 1]
    assert all(u["data"]["llm_debug"]["conversation_mode"] == "game_session" for u in updates)
    assert "previous_response_id" not in calls[0].kwargs
    assert calls[1].kwargs["previous_response_id"] == "response-1"
    assert updates[0]["data"]["commentary"] in json.dumps(calls[1].kwargs["input"])
    assert "previous_response_id" not in calls[2].kwargs
    assert calls[3].kwargs["previous_response_id"] == "response-3"
    assert service._provider is backend


def test_rejected_output_is_not_carried_to_next_move(monkeypatch):
    configure(monkeypatch)
    backend = provider(['{"text":"INVENTED_WIN"}',
                        '{"main":[{"id":0,"form":"state"}],"alternative":[]}'])
    service = AdvancedCommentService(provider=backend)
    game = state("rejected")
    asyncio.run(pipeline.GameAnnotationPipeline().run_llm_phases(game, service))
    calls = backend._client.responses.create.call_args_list
    assert len(calls) == 2
    assert "previous_response_id" not in calls[1].kwargs
    assert "INVENTED_WIN" not in json.dumps(calls[1].kwargs)


def test_cancellation_closes_game_session(monkeypatch):
    configure(monkeypatch)
    service = AdvancedCommentService(provider=provider())
    seen = []

    async def interrupt(self, run, *args):
        seen.append(run.service._provider)
        raise asyncio.CancelledError()

    monkeypatch.setattr(pipeline.GameAnnotationPipeline, "_comment_one_key_moment", interrupt)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pipeline.GameAnnotationPipeline().run_llm_phases(state("cancelled"), service))
    assert seen[0]._closed


def test_session_request_chain_is_present_in_opt_in_logs(monkeypatch, tmp_path):
    configure(monkeypatch)
    monkeypatch.setenv("LOG_LLM_TO_FILE", "1")
    monkeypatch.setenv("LOG_LLM_DIR", str(tmp_path))
    service = AdvancedCommentService(provider=provider())
    asyncio.run(pipeline.GameAnnotationPipeline().run_llm_phases(state("logged"), service))
    records = [json.loads(line) for line in (tmp_path / "logged.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[1]["session_context"]["request"]["request"]["previous_response_id"] == "response-1"
    assert records[1]["session_context"]["session_id"] == "logged"
    assert records[1]["prompt_size_scope"].startswith("current_move_only")
