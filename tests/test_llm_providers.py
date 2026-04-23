"""Provider HTTP behavior with mocked SDK clients."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.commentary.llm_providers import (
    AnthropicProvider,
    DYNAMIC_SECTION_SENTINEL,
    OpenAIProvider,
    make_llm_provider,
)
from app.core.commentary.advanced_comment_service import COMPOSER_OUTPUT_SCHEMA


class TestMakeLlmProvider(unittest.TestCase):
    def test_defaults_openai(self) -> None:
        p = make_llm_provider("openai")
        self.assertEqual(p.name, "openai")

    def test_anthropic_name(self) -> None:
        p = make_llm_provider("anthropic")
        self.assertEqual(p.name, "anthropic")


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
                schema=COMPOSER_OUTPUT_SCHEMA,
                schema_name="chess_commentary_composer",
                max_output_tokens=100,
            )
            self.assertIn("named_motifs", raw)
            self.assertEqual(n, 7)

        asyncio.run(_run())


class TestAnthropicProvider(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.env = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=False)
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
                schema=COMPOSER_OUTPUT_SCHEMA,
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
