from __future__ import annotations

import logging
import os
import time
from typing import Any

from app.core.commentary.llm_call_log import log_call as log_llm_call
from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.llm_providers import (
    LlmProvider,
    make_llm_provider,
)

logger = logging.getLogger(__name__)


def _log_llm_prompts_enabled() -> bool:
    return os.environ.get("LOG_LLM_PROMPTS", "").strip().lower() in ("1", "true")


def _debug_log_prompt(name: str, system: str, user: str) -> None:
    """Full system + user text for dev tracing (LOG_LLM_PROMPTS). No truncation."""
    if not _log_llm_prompts_enabled():
        return
    logger.info(
        "LLM_DEBUG_PROMPT name=%s\n---\nSYSTEM:\n%s\n---\nUSER:\n%s",
        name,
        system,
        user,
    )


class AdvancedCommentService:
    def __init__(
        self,
        *,
        provider: LlmProvider | None = None,
        provider_key: str | None = None,
    ) -> None:
        self._provider: LlmProvider = provider or make_llm_provider(provider_key)
        self._last_token_usage: int = 0
        self._last_llm_log_seq: int | None = None

    @property
    def provider_name(self) -> str:
        return self._provider.name

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    async def _llm_call_json_schema(
        self,
        system_prompt: str,
        user_text: str,
        *,
        model: str | None,
        effort: str,
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: int | None = None,
    ) -> str:
        """Structured JSON via provider json_schema call."""
        if _log_llm_prompts_enabled():
            _debug_log_prompt(f"json_schema:{schema_name}", system_prompt, user_text)
        resolved_model = model or resolve_model(self.provider_name, "composer")
        t0 = time.perf_counter()
        raw: str | None = None
        usage: int | None = None
        err: str | None = None
        ok = False
        try:
            raw, usage = await self._provider.json_schema_call(
                system_prompt,
                user_text,
                model=resolved_model,
                effort=effort,
                schema=schema,
                schema_name=schema_name,
                max_output_tokens=max_output_tokens,
            )
            self._last_token_usage = usage
            logger.info("LLM JSON pass %s token_usage≈%s", schema_name, usage)
            ok = True
            return (raw or "").strip()
        except Exception as e:
            err = repr(e)
            raise
        finally:
            self._log_completed_call(
                schema_name=schema_name,
                system=system_prompt,
                user=user_text,
                raw=raw,
                model=resolved_model,
                effort=effort,
                usage=usage,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                ok=ok,
                err=err,
            )

    @staticmethod
    def _log_completed_call(
        *,
        schema_name: str,
        system: str,
        user: str,
        raw: str | None,
        model: str,
        effort: str,
        usage: int | None,
        elapsed_ms: float,
        ok: bool,
        err: str | None,
    ) -> None:
        if ok and usage is None:
            logger.warning(
                "LLM json_schema call completed without usage metadata "
                "(schema_name=%s model=%s)",
                schema_name,
                model,
            )
        log_llm_call(
            pass_name=f"json_schema:{schema_name}",
            system=system,
            user=user,
            response=raw or "",
            model=model,
            effort=effort,
            schema_name=schema_name,
            token_usage=usage,
            elapsed_ms=elapsed_ms,
            ok=ok,
            error=err,
        )
