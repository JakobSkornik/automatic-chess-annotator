"""Provider HTTP behavior with mocked SDK clients."""

from __future__ import annotations

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.commentary.llm_providers import (
    DYNAMIC_SECTION_SENTINEL,
    AnthropicProvider,
    CursorProvider,
    OpenAIProvider,
    make_llm_provider,
)
from app.core.commentary.phases.composer import SINGLE_LEVEL_SCHEMA


class TestMakeLlmProvider(unittest.TestCase):
    def test_defaults_openai(self) -> None:
        p = make_llm_provider("openai")
        self.assertEqual(p.name, "openai")

    def test_anthropic_name(self) -> None:
        p = make_llm_provider("anthropic")
        self.assertEqual(p.name, "anthropic")

    def test_cursor_name(self) -> None:
        p = make_llm_provider("cursor")
        self.assertEqual(p.name, "cursor")


class TestCursorProvider(unittest.TestCase):
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

    def test_strips_sentinel(self) -> None:
        p = OpenAIProvider()
        fake_resp = MagicMock()
        fake_resp.output_text = '{"x":1}'
        fake_resp.usage = MagicMock(total_tokens=5)
        p._client = MagicMock()
        p._client.responses.create = AsyncMock(return_value=fake_resp)

        import asyncio

        async def _run() -> None:
            u = "static" + DYNAMIC_SECTION_SENTINEL + "dynamic"
            text, n = await p.text_call("sys", u, model="gpt-4.1-mini", effort="low")
            self.assertEqual(text, '{"x":1}')
            self.assertEqual(n, 5)
            call_kw = p._client.responses.create.call_args.kwargs
            user_block = call_kw["input"][1]["content"][0]["text"]
            self.assertNotIn("===DYNAMIC===", user_block)

        asyncio.run(_run())

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


class TestAnthropicProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.env = patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        super().tearDown()

    def test_json_schema_tool_use(self) -> None:
        p = AnthropicProvider()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "chess_commentary_composer"
        tool_block.input = {"named_motifs": ["pin"], "text": "Hello"}
        msg = MagicMock()
        msg.content = [tool_block]
        msg.usage = MagicMock(input_tokens=3, output_tokens=4)
        p._client = MagicMock()
        p._client.messages.create = AsyncMock(return_value=msg)

        import asyncio

        async def _run() -> None:
            raw, n = await p.json_schema_call(
                "sys",
                "static" + DYNAMIC_SECTION_SENTINEL + "tail",
                model="claude-haiku-4-5",
                effort="low",
                schema=SINGLE_LEVEL_SCHEMA,
                schema_name="chess_commentary_composer",
                max_output_tokens=256,
            )
            obj = json.loads(raw)
            self.assertEqual(obj["text"], "Hello")
            self.assertEqual(obj["named_motifs"], ["pin"])
            self.assertEqual(n, 7)
            kw = p._client.messages.create.call_args.kwargs
            self.assertIn("tools", kw)
            self.assertEqual(kw["tool_choice"]["name"], "chess_commentary_composer")

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
