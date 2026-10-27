"""Shared request capabilities, output budgets and bounded truncation recovery."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
MAX_RETRY_OUTPUT_TOKENS = 65536


def openai_reasoning_model(model: str) -> bool:
    return model.strip().lower().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def output_token_budget(model: str, effort: str, explicit: int | None = None) -> int | None:
    """Choose the initial budget; a recorded truncation retry may raise it."""
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("max_output_tokens must be positive")
        return explicit
    if openai_reasoning_model(model):
        return {"none": 8192, "minimal": 8192, "low": 8192, "medium": 16384,
                "high": 32768, "xhigh": 65536, "max": 65536}.get(effort, 16384)
    return None


def usage_breakdown(response: Any) -> dict[str, int]:
    """Provider-reported counts only: absent values remain absent, never estimated."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    values = {}
    for key in ("input_tokens", "output_tokens", "total_tokens",
                "cache_read_input_tokens", "cache_creation_input_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            values[key] = value
    for parent, child, label in (("input_tokens_details", "cached_tokens", "cached_tokens"),
                                 ("output_tokens_details", "reasoning_tokens", "reasoning_tokens")):
        value = getattr(getattr(usage, parent, None), child, None)
        if isinstance(value, int):
            values[label] = value
    return values


class OutputTruncatedError(RuntimeError):
    """No complete structured output after the bounded truncation retry."""

    def __init__(self, provider: str, usage: int) -> None:
        super().__init__(f"{provider} output truncated at token limit")
        self.token_usage = usage


async def call_with_truncation_retry(create, kwargs: dict, *, provider: str,
                                     usage_total, trace: dict) -> tuple[Any, int]:
    """Retry only token-limit truncation once, same prompt and accepted chain id.

    The retry doubles the actual cap up to 65536. Explicit caps remain the first
    attempt's cap. Record every cap/stop reason and summed usage for reproducibility.
    No response id from an incomplete attempt is ever used as conversation input.
    """
    token_key = "max_output_tokens"
    request = dict(kwargs)
    attempts = trace.setdefault("attempts", [])
    total = 0
    retry_setting = os.environ.get("COMMENTARY_TRUNCATION_RETRIES", "1")
    if retry_setting not in {"0", "1"}:
        raise ValueError("COMMENTARY_TRUNCATION_RETRIES must be 0 or 1")
    retries = int(retry_setting)
    for attempt in range(retries + 1):
        trace["request"] = request
        response = await create(**request)
        total += usage_total(response)
        details = getattr(response, "incomplete_details", None)
        reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
        truncated = getattr(response, "status", None) == "incomplete" and reason == "max_output_tokens"
        attempts.append({"max_output_tokens": request.get(token_key),
                         "stop_reason": reason if isinstance(reason, str) else None,
                         "truncated": truncated,
                         "usage": usage_breakdown(response),
                         "response_model": getattr(response, "model", None)
                             if isinstance(getattr(response, "model", None), str) else None,
                         "status": getattr(response, "status", None)
                             if isinstance(getattr(response, "status", None), str)
                             else reason if isinstance(reason, str) else None,
                         "transport": "create"})
        trace["token_usage"] = total
        trace["truncation_retries"] = attempt
        if not truncated:
            return response, total
        cap = request.get(token_key)
        logger.warning("%s token-limit truncation: cap=%s attempt=%s", provider, cap, attempt + 1)
        if attempt >= retries or not isinstance(cap, int) or cap >= MAX_RETRY_OUTPUT_TOKENS:
            logger.warning("%s truncation exhausted: token_usage=%s attempts=%s",
                           provider, total, attempts)
            raise OutputTruncatedError(provider, total)
        request = dict(request, **{token_key: min(cap * 2, MAX_RETRY_OUTPUT_TOKENS)})
    raise AssertionError("unreachable")
