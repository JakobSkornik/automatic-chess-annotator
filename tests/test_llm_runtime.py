"""Modern provider request contracts and bounded token-limit recovery."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from app.core.commentary.commentary_session import CommentarySession
from app.core.commentary.llm_providers import OpenAIProvider, cursor_model_selection, cursor_workspace
from app.core.commentary.llm_runtime import OutputTruncatedError, output_token_budget

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"},
          "ids": {"type": "array", "items": {"type": "integer"}, "maxItems": 12}},
          "required": ["text", "ids"], "additionalProperties": False}


def result(*, truncated=False, rid="r1"):
    return NS(id=rid, output_text='{"text":"ok","ids":[]}',
              status="incomplete" if truncated else "completed",
              incomplete_details=NS(reason="max_output_tokens") if truncated else None,
              usage=NS(total_tokens=7))


def backend(responses):
    value = OpenAIProvider.__new__(OpenAIProvider)
    create = AsyncMock(side_effect=responses)
    value._client = NS(responses=NS(create=create))
    return value, create


async def invoke(target, model, effort="low", cap=None):
    return await target.json_schema_call("system", "facts", model=model, effort=effort,
                                         schema=SCHEMA, schema_name="comment", max_output_tokens=cap)


@pytest.mark.parametrize("session", [False, True])
@pytest.mark.parametrize("effort,budget", [("low",8192), ("medium",16384),
                                          ("high",32768), ("xhigh",65536), ("max",65536)])
def test_modern_request_honors_effort_and_budget(effort, budget, session):
    p, create = backend([result()])
    target = CommentarySession(p, "game") if session else p
    raw, usage = asyncio.run(invoke(target, "gpt-6-luna", effort))
    request = create.call_args.kwargs
    assert '"text":"ok"' in raw and usage == 7
    assert request["reasoning"] == {"effort": effort}
    assert request["max_output_tokens"] == budget


@pytest.mark.parametrize("session", [False, True])
def test_truncation_retries_once_with_sum_and_explicit_initial_cap(session, monkeypatch):
    monkeypatch.setenv("COMMENTARY_TRUNCATION_RETRIES", "1")
    p, create = backend([result(truncated=True, rid="failed"), result(rid="success")])
    target = CommentarySession(p, "game") if session else p
    raw, usage = asyncio.run(invoke(target, "gpt-6-luna", cap=100))
    assert [c.kwargs["max_output_tokens"] for c in create.call_args_list] == [100,200]
    assert usage == 14
    assert target.last_request["truncation_retries"] == 1
    assert target.last_request["attempts"][0]["truncated"] is True
    assert all("previous_response_id" not in c.kwargs for c in create.call_args_list)
    if session:
        assert target._response_id == "success"


def test_truncation_can_be_disabled_and_never_returns_partial_json(monkeypatch):
    monkeypatch.setenv("COMMENTARY_TRUNCATION_RETRIES", "0")
    p, create = backend([result(truncated=True)])
    with pytest.raises(OutputTruncatedError) as caught:
        asyncio.run(invoke(p, "gpt-6-luna", cap=100))
    assert create.await_count == 1
    assert caught.value.token_usage == 7


def test_retry_never_chains_failed_id_and_exhaustion_resets_session(monkeypatch):
    monkeypatch.setenv("COMMENTARY_TRUNCATION_RETRIES", "1")
    p, create = backend([result(rid="accepted"), result(truncated=True, rid="bad1"),
                         result(truncated=True, rid="bad2"), result(rid="recovered")])
    session = CommentarySession(p,"game")
    async def run():
        await invoke(session,"gpt-6-luna",cap=100)
        await session.record_annotation(dict(text="accepted",rendering="llm",contract_ok=True,ply=1))
        with pytest.raises(OutputTruncatedError):
            await invoke(session,"gpt-6-luna",cap=100)
        assert session._response_id is None and not session._pending_raw
        await invoke(session,"gpt-6-luna",cap=100)
    asyncio.run(run())
    calls = create.call_args_list
    assert calls[1].kwargs["previous_response_id"] == "accepted"
    assert calls[2].kwargs["previous_response_id"] == "accepted"
    assert "previous_response_id" not in calls[3].kwargs
    assert "accepted" in str(calls[3].kwargs["input"])


def test_budget_cap_retry_upper_bound(monkeypatch):
    monkeypatch.setenv("COMMENTARY_TRUNCATION_RETRIES", "1")
    p, create = backend([result(truncated=True)])
    with pytest.raises(OutputTruncatedError):
        asyncio.run(invoke(p,"gpt-6-luna",cap=65536))
    assert create.await_count == 1
    assert output_token_budget("gpt-6-luna","high",1200) == 1200


def test_retry_trace_retains_usage_details(monkeypatch):
    monkeypatch.setenv("COMMENTARY_TRUNCATION_RETRIES", "1")
    response = result()
    response.usage = NS(total_tokens=7,input_tokens=3,output_tokens=4,
                        input_tokens_details=NS(cached_tokens=2),
                        output_tokens_details=NS(reasoning_tokens=3))
    p, _ = backend([response])
    asyncio.run(invoke(p,"gpt-6-luna"))
    assert p.last_request["attempts"][0]["usage"] == dict(total_tokens=7,
        input_tokens=3,output_tokens=4,cached_tokens=2,reasoning_tokens=3)


def test_cursor_effort_is_sent_only_to_models_that_expose_it():
    assert cursor_model_selection("grok-4.6", "medium") == {
        "id": "grok-4.6", "params": [{"id": "effort", "value": "medium"}]}
    assert cursor_model_selection("composer-2.5", "medium") == "composer-2.5"
    assert cursor_model_selection("grok-4.6", "max") == "grok-4.6"  # Not offered for Grok 4.6.


def test_cursor_workspace_is_outside_the_repository(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("CURSOR_AGENT_WORKSPACE", str(tmp_path / "agent"))
    workspace = cursor_workspace()
    assert workspace.is_dir() and not any(workspace.iterdir())
    repo = Path(__file__).resolve().parents[1]
    assert repo not in workspace.parents and workspace != repo
