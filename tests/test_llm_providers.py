"""Provider HTTP behavior with mocked SDK clients."""

from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.commentary.llm_providers import (
    DYNAMIC_SECTION_SENTINEL,
    CursorProvider,
    OpenAIProvider,
    make_llm_provider,
)
from app.core.commentary.phases.composer import SINGLE_LEVEL_SCHEMA


class TestMakeLlmProvider(unittest.TestCase):
    def test_defaults_openai(self) -> None:
        p = make_llm_provider("openai")
        self.assertEqual(p.name, "openai")

    def test_anthropic_is_no_longer_a_provider(self) -> None:
        self.assertEqual(make_llm_provider("anthropic").name, "openai")

    def test_grok_uses_cursor_transport_with_its_own_policy(self) -> None:
        from app.core.commentary.advanced_comment_service import AdvancedCommentService
        from app.core.commentary.llm_policy import resolve_model
        p = make_llm_provider("grok")
        self.assertEqual((p.name, p.policy_key), ("cursor", "grok"))
        service = AdvancedCommentService(provider=p)
        self.assertEqual(resolve_model(service.provider_name, "composer"), "grok-4.6")
        self.assertEqual(resolve_model("cursor", "composer"), "composer-2.5")

    def test_cursor_name(self) -> None:
        p = make_llm_provider("cursor")
        self.assertEqual(p.name, "cursor")


class TestCursorProvider(unittest.TestCase):
    def test_transient_internal_200_retries_and_recovers(self):
        import asyncio
        error = RuntimeError("internal")
        error.code, error.status = "internal", 200
        self.mock_agent.prompt.side_effect = [error, SimpleNamespace(status="finished", result="OK", usage=None)]
        with patch("app.core.commentary.llm_providers.asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.assertEqual(asyncio.run(CursorProvider()._prompt("test", model="composer-2.5"))[0], "OK")
            sleep.assert_awaited_once_with(1)
        self.assertEqual(self.mock_agent.prompt.call_count, 2)

    def test_retry_exhaustion_and_auth_errors(self):
        import asyncio
        error = RuntimeError("internal")
        error.code, error.status = "internal", 200
        self.mock_agent.prompt.side_effect = error
        with patch("app.core.commentary.llm_providers.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(RuntimeError):
                asyncio.run(CursorProvider()._prompt("test", model="composer-2.5"))
        self.assertEqual(self.mock_agent.prompt.call_count, 3)
        self.mock_agent.prompt.reset_mock()
        error.code, error.status = "unauthenticated", 401
        with self.assertRaises(RuntimeError):
            asyncio.run(CursorProvider()._prompt("test", model="composer-2.5"))
        self.assertEqual(self.mock_agent.prompt.call_count, 1)

    def setUp(self) -> None:
        super().setUp()
        self.env = patch.dict(
            "os.environ",
            {"CURSOR_API_KEY": "cursor_test_key"},
            clear=False,
        )
        self.env.start()
        self.mock_agent = MagicMock()
        self.mock_opts = MagicMock(side_effect=lambda **kw: kw)
        self.mock_local = MagicMock(side_effect=lambda **kw: kw)
        self.mock_sdk = MagicMock(
            Agent=self.mock_agent,
            AgentOptions=self.mock_opts,
            LocalAgentOptions=self.mock_local,
            CursorAgentError=Exception,
        )
        self.sdk_mod = patch.dict(sys.modules, {"cursor_sdk": self.mock_sdk})
        self.sdk_mod.start()

    def tearDown(self) -> None:
        self.sdk_mod.stop()
        self.env.stop()
        super().tearDown()

    def test_json_schema_call_strips_sentinel_and_returns_json(self) -> None:
        payload = (
            '{"named_motifs":[],"text":"ok","better_alternative":"",'
            '"rag_idea_used":"","rag_applied":false}'
        )
        self.mock_agent.prompt.return_value = SimpleNamespace(
            status="finished",
            result=payload,
            usage=None,
        )
        p = CursorProvider()

        import asyncio

        async def _run() -> None:
            raw, n = await p.json_schema_call(
                "sys",
                "static" + DYNAMIC_SECTION_SENTINEL + "dynamic",
                model="composer-2.5",
                effort="low",
                schema=SINGLE_LEVEL_SCHEMA,
                schema_name="chess_commentary_composer",
                max_output_tokens=100,
            )
            self.assertIn("named_motifs", raw)
            self.assertEqual(n, 0)
            prompt_arg = self.mock_agent.prompt.call_args[0][0]
            self.assertIn("chess_commentary_composer", prompt_arg)
            self.assertNotIn("===DYNAMIC===", prompt_arg)
            self.assertIn("dynamic", prompt_arg)

        asyncio.run(_run())


class TestOpenAIProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.env = patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        super().tearDown()

    def test_json_schema_call(self) -> None:
        p = OpenAIProvider()
        fake_resp = MagicMock()
        fake_resp.output_text = '{"named_motifs":[],"text":"hi"}'
        fake_resp.usage = MagicMock(total_tokens=7)
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(return_value=fake_resp)

        import asyncio

        async def _run() -> None:
            raw, n = await p.json_schema_call(
                "sys",
                "user",
                model="gpt-4.1-mini",
                effort="low",
                schema=SINGLE_LEVEL_SCHEMA,
                schema_name="chess_commentary_composer",
                max_output_tokens=100,
            )
            self.assertIn("named_motifs", raw)
            self.assertEqual(n, 7)

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
