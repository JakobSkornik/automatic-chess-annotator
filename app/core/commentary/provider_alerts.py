"""Remember providers whose last call failed for lack of credits, so the UI can say so.

Comment generation falls back to template text on any provider error, which hides an
empty account. Credit failures are kept in memory per provider until a call succeeds.
"""
import threading
import time
from typing import Any

_CREDIT_CODES = {"insufficient_quota", "billing_hard_limit_reached", "payment_required", "billing_not_active"}
_CREDIT_PHRASES = ("insufficient_quota", "exceeded your current quota", "billing", "credit balance",
                   "out of credits", "insufficient credits", "usage limit", "spend limit", "spending limit",
                   "payment required", "quota exceeded")

_lock = threading.Lock()
_alerts: dict[str, dict[str, Any]] = {}


def is_credit_error(exc: BaseException) -> bool:
    """OpenAI reports code insufficient_quota (HTTP 429); other providers use 402 or a billing message."""
    body = getattr(exc, "body", None)
    codes = {str(getattr(exc, "code", "") or "").lower()}
    if isinstance(body, dict):
        nested = body.get("error") if isinstance(body.get("error"), dict) else body
        codes |= {str(nested.get("code") or "").lower(), str(nested.get("type") or "").lower()}
    if codes & _CREDIT_CODES or getattr(exc, "status", None) == 402 or getattr(exc, "status_code", None) == 402:
        return True
    text = str(exc).lower()
    return any(phrase in text for phrase in _CREDIT_PHRASES)


def record_failure(provider: str, exc: BaseException) -> None:
    if not is_credit_error(exc):
        return
    with _lock:
        _alerts[provider] = {"provider": provider, "kind": "credits", "since": _alerts.get(provider, {}).get("since", time.time()),
                             "last_seen": time.time(), "message": str(exc)[:300]}


def record_success(provider: str) -> None:
    with _lock:
        _alerts.pop(provider, None)


def current() -> list[dict[str, Any]]:
    with _lock:
        return [dict(a) for a in _alerts.values()]
