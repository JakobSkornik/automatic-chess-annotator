"""LLM provider abstraction: OpenAI, Anthropic, and Cursor SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, Type, runtime_checkable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Lazy-import targets for CursorProvider (patchable in tests).
_Agent: Any = None
_AgentOptions: Any = None
_LocalAgentOptions: Any = None
_CursorAgentError: Type[BaseException] = Exception

DYNAMIC_SECTION_SENTINEL = "\n\n===DYNAMIC===\n\n"


def _read_bridge_discovery_threaded(
    process: subprocess.Popen[str], timeout: float
) -> Mapping[str, Any]:
    """Windows/Python<3.12: avoid os.set_blocking + os.read on subprocess stderr."""
    from cursor_sdk._bridge import parse_discovery_line
    from cursor_sdk.errors import CursorSDKError

    if process.stderr is None:
        raise CursorSDKError("Bridge process stderr is unavailable")

    line_queue: queue.Queue[str | None] = queue.Queue()
    read_error: list[BaseException] = []

    def _reader() -> None:
        try:
            assert process.stderr is not None
            for line in process.stderr:
                line_queue.put(line)
        except Exception as exc:
            read_error.append(exc)
        finally:
            line_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    deadline = time.monotonic() + timeout
    stderr_lines: list[str] = []

    while time.monotonic() < deadline:
        if read_error:
            raise CursorSDKError("Bridge stderr read failed") from read_error[0]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            line = line_queue.get(timeout=min(0.1, remaining))
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        if line is None:
            break
        stderr_lines.append(line)
        discovery = parse_discovery_line(line)
        if discovery is not None:
            return discovery

    exit_code = process.poll()
    if exit_code is not None:
        raise CursorSDKError(
            f"Bridge exited before discovery with status {exit_code}: "
            + "".join(stderr_lines)
        )
    raise CursorSDKError("Timed out waiting for bridge discovery")


def _patch_cursor_bridge_discovery() -> None:
    """Replace cursor-sdk bridge discovery on Windows Python versions before 3.12."""
    if sys.platform != "win32" or sys.version_info >= (3, 12):
        return
    try:
        import cursor_sdk._bridge as bridge_mod
    except (ImportError, ModuleNotFoundError):
        return
    if not hasattr(bridge_mod, "_read_discovery"):
        return
    bridge_mod._read_discovery = _read_bridge_discovery_threaded  # type: ignore[attr-defined]


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


def _strip_json_fence(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _usage_total_cursor(result: Any) -> int:
    u = getattr(result, "usage", None)
    if u is None:
        return 0
    total = getattr(u, "total_tokens", None)
    if isinstance(total, int):
        return total
    inp = getattr(u, "input_tokens", None) or 0
    out = getattr(u, "output_tokens", None) or 0
    try:
        return int(inp) + int(out)
    except (TypeError, ValueError):
        return 0


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


class CursorProvider:
    """Cursor SDK one-shot agent (Composer-2.5) via Agent.prompt in a worker thread."""

    def __init__(self) -> None:
        global _Agent, _AgentOptions, _LocalAgentOptions, _CursorAgentError
        try:
            from cursor_sdk import Agent, AgentOptions, CursorAgentError, LocalAgentOptions
        except ImportError as e:
            raise ImportError("Install the cursor-sdk package to use CursorProvider") from e
        _patch_cursor_bridge_discovery()
        _Agent = Agent
        _AgentOptions = AgentOptions
        _LocalAgentOptions = LocalAgentOptions
        _CursorAgentError = CursorAgentError

    @property
    def name(self) -> str:
        return "cursor"

    def is_configured(self) -> bool:
        return bool(os.environ.get("CURSOR_API_KEY", "").strip())

    def _agent_options(self, model: str) -> Any:
        api_key = os.environ.get("CURSOR_API_KEY", "").strip()
        return _AgentOptions(
            api_key=api_key,
            model=model,
            local=_LocalAgentOptions(cwd=os.getcwd()),
        )

    async def _prompt(self, prompt: str, *, model: str) -> Tuple[str, int]:
        """Run Agent.prompt off the event loop. effort/max_output_tokens are ignored."""
        opts = self._agent_options(model)

        def _run() -> Any:
            return _Agent.prompt(prompt, opts)

        try:
            result = await asyncio.to_thread(_run)
        except _CursorAgentError:
            raise
        usage = _usage_total_cursor(result)
        status = getattr(result, "status", None)
        if status == "error":
            rid = getattr(result, "id", "?")
            logger.warning("Cursor Agent.prompt run failed: id=%s", rid)
            return "", usage
        text = getattr(result, "result", None) or ""
        return str(text).strip(), usage

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
        prompt = f"{system.strip()}\n\n{user.strip()}".strip()
        return await self._prompt(prompt, model=model)

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
        schema_hint = (
            f"Return ONLY a JSON object matching schema '{schema_name}'. "
            "No markdown fences, no prose outside JSON.\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        prompt = f"{system.strip()}\n\n{schema_hint}\n\n{user.strip()}".strip()
        raw, usage = await self._prompt(prompt, model=model)
        return _strip_json_fence(raw), usage


def make_llm_provider(provider_key: Optional[str] = None) -> LlmProvider:
    k = (provider_key or os.environ.get("LLM_DEFAULT_PROVIDER") or "openai").strip().lower()
    if k == "anthropic":
        return AnthropicProvider()
    if k == "cursor":
        return CursorProvider()
    return OpenAIProvider()
