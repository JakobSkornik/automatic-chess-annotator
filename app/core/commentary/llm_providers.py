"""LLM provider abstraction: OpenAI Responses API vs Anthropic Messages API."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DYNAMIC_SECTION_SENTINEL = "\n\n===DYNAMIC===\n\n"


def _openai_model_uses_reasoning_api(model_id: str) -> bool:
    m = (model_id or "").strip().lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )


def _strip_sentinel_for_openai(user_text: str) -> str:
    if DYNAMIC_SECTION_SENTINEL not in user_text:
        return user_text
    return user_text.replace(DYNAMIC_SECTION_SENTINEL, "\n\n")


def _usage_total_openai(response: Any) -> int:
    u = getattr(response, "usage", None)
    if u is None:
        return 0
    total = getattr(u, "total_tokens", None)
    if isinstance(total, int):
        return total
    return 0


def _usage_total_anthropic(msg: Any) -> int:
    u = getattr(msg, "usage", None)
    if u is None:
        return 0
    inp = getattr(u, "input_tokens", None) or 0
    out = getattr(u, "output_tokens", None) or 0
    try:
        return int(inp) + int(out)
    except (TypeError, ValueError):
        return 0


@runtime_checkable
class LlmProvider(Protocol):
    @property
    def name(self) -> str: ...

    def is_configured(self) -> bool: ...

    async def text_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int]: ...

    async def json_schema_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        schema: Dict[str, Any],
        schema_name: str,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int]: ...


class OpenAIProvider:
    def __init__(self) -> None:
        self._client = AsyncOpenAI()

    @property
    def name(self) -> str:
        return "openai"

    def is_configured(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())

    async def text_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        user = _strip_sentinel_for_openai(user)
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
        }
        if _openai_model_uses_reasoning_api(model):
            kwargs["reasoning"] = {"effort": effort or "low"}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        response = await self._client.responses.create(**kwargs)
        text = ""
        if hasattr(response, "output_text") and response.output_text:
            text = response.output_text
        elif hasattr(response, "output") and response.output:
            try:
                text = response.output[0].content[0].text
            except Exception:
                text = ""
        return text, _usage_total_openai(response)

    async def json_schema_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        schema: Dict[str, Any],
        schema_name: str,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        user = _strip_sentinel_for_openai(user)
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system}]},
                {"role": "user", "content": [{"type": "input_text", "text": user}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if _openai_model_uses_reasoning_api(model):
            kwargs["reasoning"] = {"effort": effort}
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        response = await self._client.responses.create(**kwargs)
        raw = ""
        if hasattr(response, "output_text") and response.output_text:
            raw = response.output_text.strip()
        elif hasattr(response, "output") and response.output:
            try:
                raw = response.output[0].content[0].text.strip()
            except Exception:
                raw = ""
        return raw, _usage_total_openai(response)


class AnthropicProvider:
    """Anthropic Messages API with optional prompt caching on the static user prefix."""

    def __init__(self) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("Install the anthropic package to use AnthropicProvider") from e
        self._client = AsyncAnthropic()

    @property
    def name(self) -> str:
        return "anthropic"

    def is_configured(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

    def _user_content_blocks(self, user: str) -> Any:
        if DYNAMIC_SECTION_SENTINEL in user:
            static, dynamic = user.split(DYNAMIC_SECTION_SENTINEL, 1)
            return [
                {
                    "type": "text",
                    "text": static.strip(),
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": dynamic.strip()},
            ]
        return user

    async def text_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        max_tok = max_output_tokens if max_output_tokens is not None else 4096
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tok,
            "system": system,
            "messages": [{"role": "user", "content": self._user_content_blocks(user)}],
        }
        msg = await self._client.messages.create(**kwargs)
        parts: list[str] = []
        for block in getattr(msg, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                parts.append(getattr(block, "text", "") or "")
        return "".join(parts).strip(), _usage_total_anthropic(msg)

    async def json_schema_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        schema: Dict[str, Any],
        schema_name: str,
        max_output_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        tool_name = schema_name or "structured_output"
        tools = [
            {
                "name": tool_name,
                "description": "Emit structured JSON matching the requested schema.",
                "input_schema": schema,
            }
        ]
        max_tok = max_output_tokens if max_output_tokens is not None else 4096
        user_payload = self._user_content_blocks(user)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tok,
            "system": system,
            "tools": tools,
            "tool_choice": {"type": "tool", "name": tool_name},
            "messages": [{"role": "user", "content": user_payload}],
        }
        msg = await self._client.messages.create(**kwargs)
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == tool_name:
                inp = getattr(block, "input", None)
                if isinstance(inp, dict):
                    return json.dumps(inp), _usage_total_anthropic(msg)
                if isinstance(inp, str):
                    return inp, _usage_total_anthropic(msg)
        logger.warning("Anthropic json_schema_call: no matching tool_use in response")
        return "", _usage_total_anthropic(msg)


def make_llm_provider(provider_key: Optional[str] = None) -> LlmProvider:
    k = (provider_key or os.environ.get("LLM_DEFAULT_PROVIDER") or "openai").strip().lower()
    if k == "anthropic":
        return AnthropicProvider()
    return OpenAIProvider()
