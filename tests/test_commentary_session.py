"""Mocked provider tests for per-game commentary sessions."""

from __future__ import annotations

import asyncio
import json
import threading
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.commentary.commentary_session import CommentarySession


SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}},
          "required": ["text"], "additionalProperties": False}


def response(rid: str, text: str = '{"text":"draft"}') -> Any:
    return SimpleNamespace(id=rid, output_text=text,
                           usage=SimpleNamespace(total_tokens=5))


def provider(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, is_configured=lambda: True)


async def call(session: CommentarySession, user: str = "move facts") -> tuple[str, int]:
    return await session.json_schema_call(
        "commentary instructions", user, model="gpt-4.1-mini", effort="low",
        schema=SCHEMA, schema_name="commentary", max_output_tokens=100,
    )


async def commit(session: CommentarySession, *, text: str = "accepted",
                 rendering: str = "llm", contract_ok: bool = True,
                 ply: int = 1, san: str = "e4") -> None:
    await session.record_annotation({"ply": ply, "san": san, "fen_after": f"fen-{ply}",
                                     "text": text, "rendering": rendering,
                                     "contract_ok": contract_ok})


class CommentarySessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_uses_native_chain_and_system_only_at_first_turn(self) -> None:
        p = provider("openai")
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(side_effect=[response("r1"), response("r2")])
        session = CommentarySession(p, "game-1")

        await call(session, "first move")
        first = p._client.responses.create.call_args_list[0].kwargs
        self.assertEqual([part["role"] for part in first["input"]], ["system", "user"])
        await commit(session, text="Accepted first annotation")
        await call(session, "second move")

        second = p._client.responses.create.call_args_list[1].kwargs
        self.assertEqual(second["previous_response_id"], "r1")
        self.assertEqual([part["role"] for part in second["input"]], ["user"])
        second_text = second["input"][0]["content"][0]["text"]
        self.assertIn("Accepted first annotation", second_text)
        self.assertNotIn('"request":', second_text)
        self.assertEqual(session.turn_count, 2)
        self.assertEqual(session.session_id, "game-1")
        self.assertEqual(session.last_request["provider"], "openai")

    async def test_cursor_reuses_one_agent_and_closes_it(self) -> None:
        p = provider("cursor")
        p._agent_options = lambda model, effort=None: {"model": model}
        fake_run = MagicMock()
        fake_run.wait.return_value = SimpleNamespace(
            status="finished", result='{"text":"draft"}', usage=None
        )
        agent = MagicMock()
        agent.send.return_value = fake_run
        fake_sdk_agent = MagicMock()
        fake_sdk_agent.create.return_value = agent
        session = CommentarySession(p, "cursor-game")

        with patch("app.core.commentary.llm_providers._Agent", fake_sdk_agent):
            await call(session, "first")
            await commit(session, text="Verified line")
            await call(session, "second")
            await session.close()

        fake_sdk_agent.create.assert_called_once_with({"model": "gpt-4.1-mini"})
        self.assertEqual(agent.send.call_count, 2)
        self.assertIn("commentary instructions", agent.send.call_args_list[0].args[0])
        self.assertNotIn("commentary instructions", agent.send.call_args_list[1].args[0])
        self.assertIn("Verified line", agent.send.call_args_list[1].args[0])
        agent.close.assert_called_once()

    async def test_failed_cursor_run_retries_once_on_rebuilt_agent(self) -> None:
        p = provider("cursor")
        p._agent_options = lambda model, effort=None: {"model": model}
        ok = MagicMock(); ok.wait.return_value = SimpleNamespace(status="finished", result='{"text":"draft"}', usage=None)
        bad = MagicMock(); bad.wait.return_value = SimpleNamespace(status="error", id="run-bad", result="", usage=None)
        first, second = MagicMock(), MagicMock()
        first.send.side_effect = [ok, bad]
        second.send.return_value = ok
        fake_sdk_agent = MagicMock()
        fake_sdk_agent.create.side_effect = [first, second]
        session = CommentarySession(p, "cursor-retry")

        with patch("app.core.commentary.llm_providers._Agent", fake_sdk_agent):
            await call(session, "first")
            await commit(session, text="Verified line")
            raw, _ = await call(session, "second")
            await session.close()

        self.assertIn("draft", raw)
        self.assertEqual(session.last_request["failed_runs"], ["run-bad"])
        first.close.assert_called_once()
        retry_prompt = second.send.call_args.args[0]
        self.assertIn("commentary instructions", retry_prompt)
        self.assertIn("Verified line", retry_prompt)

    async def test_second_failed_cursor_run_is_raised(self) -> None:
        from app.core.commentary.commentary_session import CursorRunFailed
        p = provider("cursor")
        p._agent_options = lambda model, effort=None: {"model": model}
        bad = MagicMock(); bad.wait.return_value = SimpleNamespace(status="error", id="run-bad", result="", usage=None)
        agent = MagicMock(); agent.send.return_value = bad
        fake_sdk_agent = MagicMock(); fake_sdk_agent.create.return_value = agent
        session = CommentarySession(p, "cursor-fail")
        with patch("app.core.commentary.llm_providers._Agent", fake_sdk_agent):
            with self.assertRaises(CursorRunFailed):
                await call(session, "first")
        self.assertEqual(fake_sdk_agent.create.call_count, 2)

    async def test_rejected_turn_resets_openai_to_canonical_history(self) -> None:
        p = provider("openai")
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(side_effect=[
            response("r1"), response("r2"), response("r3")
        ])
        session = CommentarySession(p, "rebuild-game")

        await call(session, "accepted request")
        await commit(session, text="Verified fact", ply=1)
        await call(session, "bad request")
        await commit(session, text="fallback sentence", rendering="fallback")
        await call(session, "third request")

        third = p._client.responses.create.call_args_list[2].kwargs
        self.assertNotIn("previous_response_id", third)
        payload = third["input"][1]["content"][0]["text"]
        self.assertIn("Verified fact", payload)
        self.assertIn('"request": "accepted request"', payload)
        self.assertNotIn("fallback sentence", payload)
        self.assertNotIn("bad request", payload)

    async def test_failed_provider_turn_rebuilds_from_committed_annotations(self) -> None:
        p = provider("openai")
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(side_effect=[
            response("r1"), RuntimeError("provider failed"), response("r3")
        ])
        session = CommentarySession(p, "failed-game")

        await call(session, "accepted request")
        await commit(session, text="Only committed annotation")
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            await call(session, "request that failed")
        await call(session, "request after recovery")

        recovered = p._client.responses.create.call_args_list[2].kwargs
        self.assertNotIn("previous_response_id", recovered)
        payload = recovered["input"][1]["content"][0]["text"]
        self.assertIn("Only committed annotation", payload)
        self.assertNotIn("request that failed", payload)

    async def test_provider_state_isolated_between_games(self) -> None:
        p = provider("openai")
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(side_effect=[
            response("a1"), response("b1"), response("a2")
        ])
        first = CommentarySession(p, "game-a")
        second = CommentarySession(p, "game-b")

        await call(first, "a move")
        await commit(first, text="A annotation")
        await call(second, "b move")
        await commit(second, text="B annotation")
        await call(first, "a next move")

        calls = p._client.responses.create.call_args_list
        self.assertNotIn("previous_response_id", calls[1].kwargs)
        self.assertEqual(calls[2].kwargs["previous_response_id"], "a1")
        self.assertIn("A annotation", calls[2].kwargs["input"][0]["content"][0]["text"])
        self.assertNotIn("B annotation", str(calls[2].kwargs["input"]))

    async def test_all_intervening_annotations_and_move_ids_are_delivered(self) -> None:
        p = provider("openai")
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(side_effect=[
            response("r1"), response("r2"), response("r3")
        ])
        session = CommentarySession(p, "interstitial-game")

        await call(session, "first")
        await commit(session, text="First accepted annotation", ply=1, san="e4")
        await call(session, "second")
        await commit(session, text="Second accepted annotation", ply=2, san="e5")
        await commit(session, text="Template annotation", rendering="template",
                     ply=3, san="Nf3")
        await commit(session, text="Skipped-move summary", rendering="skipped",
                     ply=4, san="Nc6")
        await call(session, "fourth")

        payload = p._client.responses.create.call_args_list[2].kwargs["input"][0]["content"][0]["text"]
        self.assertIn('"ply": 2', payload)
        self.assertIn('"san": "e5"', payload)
        self.assertIn('"fen_after": "fen-2"', payload)
        self.assertIn("Template annotation", payload)
        self.assertIn('"ply": 3', payload)
        self.assertIn("Skipped-move summary", payload)
        self.assertNotIn("verified annotation", payload.lower())

    async def test_config_is_immutable_and_closed_session_rejects_records(self) -> None:
        p = provider("openai")
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(return_value=response("r1"))
        session = CommentarySession(p, "immutable-game")
        await call(session)
        await commit(session)

        with self.assertRaisesRegex(ValueError, "immutable"):
            await session.json_schema_call(
                "changed instructions", "next", model="gpt-4.1-mini", effort="low",
                schema=SCHEMA, schema_name="commentary", max_output_tokens=100,
            )
        await session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await commit(session, text="late")

    async def test_cancelled_cursor_run_waits_for_worker_then_closes_agent(self) -> None:
        p = provider("cursor")
        p._agent_options = lambda model, effort=None: {"model": model}
        started = threading.Event()
        release = threading.Event()
        run = MagicMock()

        def wait_for_run():
            started.set()
            release.wait(timeout=5)
            return SimpleNamespace(status="finished", result='{"text":"draft"}', usage=None)

        run.wait.side_effect = wait_for_run
        agent = MagicMock()
        agent.send.return_value = run
        fake_sdk_agent = MagicMock()
        fake_sdk_agent.create.return_value = agent
        session = CommentarySession(p, "cancelled-cursor")

        with patch("app.core.commentary.llm_providers._Agent", fake_sdk_agent):
            task = asyncio.create_task(call(session))
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done(), "cancel should wait for blocking SDK worker cleanup")
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIsNone(session._cursor_agent)
            agent.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
