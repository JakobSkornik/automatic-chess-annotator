"""LLM provider abstraction: OpenAI and Cursor-hosted models (Composer, Grok) via the Cursor SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from openai import AsyncOpenAI

from app.core.commentary.json_utils import strip_json_fence
from app.core.commentary.llm_runtime import (
    openai_reasoning_model, output_token_budget, call_with_truncation_retry,
)

logger = logging.getLogger(__name__)

# Lazy-import targets for CursorProvider (patchable in tests).
_Agent: Any = None
_AgentOptions: Any = None
_LocalAgentOptions: Any = None
_CursorAgentError: type[BaseException] = Exception

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
    return openai_reasoning_model(model_id or "")


def _strip_sentinel_for_openai(user_text: str) -> str:
    if DYNAMIC_SECTION_SENTINEL not in user_text:
        return user_text
    return user_text.replace(DYNAMIC_SECTION_SENTINEL, "\n\n")


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


@runtime_checkable
class LlmProvider(Protocol):
    """A pluggable LLM backend. Every provider exposes a stable name, a
    configuration check, and a JSON-schema-constrained completion call."""

    @property
    def name(self) -> str:
        """Stable transport key (e.g. 'openai', 'cursor')."""
        ...

    def is_configured(self) -> bool:
        """True when credentials/config are present, so a call can be attempted."""
        ...

    async def json_schema_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> tuple[str, int]:
        """Run a schema-constrained completion; return (raw_text, token_usage)."""
        ...


class OpenAIProvider:
    def __init__(self) -> None:
        self._client = AsyncOpenAI()

    @property
    def name(self) -> str:
        return "openai"

    def is_configured(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY", "").strip())

    async def json_schema_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> tuple[str, int]:
        user = _strip_sentinel_for_openai(user)
        kwargs: dict[str, Any] = {
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
        budget = output_token_budget(model, effort, max_output_tokens)
        if budget is not None:
            kwargs["max_output_tokens"] = budget
        self.last_request = {"provider": self.name}
        response, usage = await call_with_truncation_retry(
            self._client.responses.create, kwargs, provider=self.name,
            usage_total=_usage_total_openai, trace=self.last_request)
        raw = ""
        if hasattr(response, "output_text") and response.output_text:
            raw = response.output_text.strip()
        elif hasattr(response, "output") and response.output:
            try:
                raw = response.output[0].content[0].text.strip()
            except Exception:
                raw = ""
        return raw, usage


# Effort parameter ids differ per Cursor-hosted model (Cursor.models.list(), 26 October 2026).
# Models without an entry (e.g. composer-2.5) expose no effort setting.
CURSOR_EFFORT_PARAMS = {"grok-4.6": "effort"}
CURSOR_EFFORT_VALUES = {"low", "medium", "high", "xhigh"}


def cursor_model_selection(model: str, effort: str | None) -> Any:
    """Model id plus its effort parameter when the model supports one."""
    param = CURSOR_EFFORT_PARAMS.get(model)
    if param is None or effort not in CURSOR_EFFORT_VALUES:
        return model
    return {"id": model, "params": [{"id": param, "value": effort}]}


def cursor_workspace() -> Path:
    """Empty working directory for Cursor agents, keeping the repository and .env out of their workspace."""
    configured = os.environ.get("CURSOR_AGENT_WORKSPACE", "").strip()
    path = Path(configured) if configured else Path(tempfile.gettempdir()) / "aca-cursor-agent"
    path.mkdir(parents=True, exist_ok=True)
    return path


class CursorProvider:
    """Cursor-hosted models (Composer, Grok) through the Cursor SDK agent.

    ``key`` selects the model policy ('cursor' = Composer, 'grok' = Grok); the
    transport is the same Cursor agent for both.
    """

    def __init__(self, key: str = "cursor") -> None:
        self.policy_key = key
        global _Agent, _AgentOptions, _LocalAgentOptions, _CursorAgentError
        try:
            from cursor_sdk import (
                Agent,
                AgentOptions,
                CursorAgentError,
                LocalAgentOptions,
            )
        except ImportError as e:
            raise ImportError(
                "Install the cursor-sdk package to use CursorProvider"
            ) from e
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

    def _agent_options(self, model: str, effort: str | None = None) -> Any:
        api_key = os.environ.get("CURSOR_API_KEY", "").strip()
        return _AgentOptions(
            api_key=api_key,
            model=cursor_model_selection(model, effort),
            # Empty workspace without user/project rules, so the repository and .env
            # are not in the agent's workspace. This is not OS-level isolation: the
            # Cursor bridge rejects sandboxing on Windows ("not supported in this environment").
            local=_LocalAgentOptions(cwd=str(cursor_workspace()), setting_sources=[]),
        )

    async def _prompt(self, prompt: str, *, model: str, effort: str | None = None) -> tuple[str, int]:
        """Run Agent.prompt off the event loop. max_output_tokens is not supported by Cursor."""
        opts = self._agent_options(model, effort)

        def _run() -> Any:
            return _Agent.prompt(prompt, opts)

        for attempt in range(3):
            try:
                result = await asyncio.to_thread(_run)
                break
            except Exception as exc:
                # The SDK can report code=internal with HTTP status 200.
                # Retry only transient provider failures, never auth/input errors.
                status = getattr(exc, "status", None)
                code = getattr(exc, "code", None)
                transient = code in ("internal", "unavailable", "resource_exhausted") or (
                    isinstance(status, int) and (status == 429 or 500 <= status < 600)
                )
                if not transient or attempt == 2:
                    raise
                logger.warning("Cursor transient failure (%s, code=%s, status=%s); retry %d/2",
                               type(exc).__name__, code, status, attempt + 1)
                await asyncio.sleep(2 ** attempt)
        usage = _usage_total_cursor(result)
        status = getattr(result, "status", None)
        if status == "error":
            rid = getattr(result, "id", "?")
            logger.warning("Cursor Agent.prompt run failed: id=%s", rid)
            return "", usage
        text = getattr(result, "result", None) or ""
        return str(text).strip(), usage

    async def json_schema_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> tuple[str, int]:
        user = _strip_sentinel_for_openai(user)
        schema_hint = (
            f"Return ONLY a JSON object matching schema '{schema_name}'. "
            "No markdown fences, no prose outside JSON.\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        prompt = f"{system.strip()}\n\n{schema_hint}\n\n{user.strip()}".strip()
        raw, usage = await self._prompt(prompt, model=model, effort=effort)
        return strip_json_fence(raw), usage


def make_llm_provider(provider_key: str | None = None) -> LlmProvider:
    """Build the configured LLM provider for ``provider_key`` (falls back to the default)."""
    k = (
        (provider_key or os.environ.get("LLM_DEFAULT_PROVIDER") or "openai")
        .strip()
        .lower()
    )
    if k in ("cursor", "grok"):
        return CursorProvider(k)
    return OpenAIProvider()
