import asyncio

import httpx
import openai
import pytest

from app.core.commentary import provider_alerts
from app.core.commentary.advanced_comment_service import AdvancedCommentService


def _openai_error(code: str, message: str) -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    return openai.RateLimitError(message, response=response, body={"message": message, "type": code, "code": code})


@pytest.fixture(autouse=True)
def _clear_alerts():
    provider_alerts._alerts.clear()
    yield
    provider_alerts._alerts.clear()


def test_credit_errors_are_recognised_and_rate_limits_are_not():
    assert provider_alerts.is_credit_error(_openai_error("insufficient_quota", "You exceeded your current quota"))
    assert provider_alerts.is_credit_error(RuntimeError("Payment required: out of credits"))
    assert not provider_alerts.is_credit_error(_openai_error("rate_limit_exceeded", "Rate limit reached for requests"))
    assert not provider_alerts.is_credit_error(ValueError("invalid JSON"))


class _Provider:
    name = "openai"

    def __init__(self, error=None):
        self.error = error

    def is_configured(self):
        return True

    async def json_schema_call(self, *args, **kwargs):
        if self.error:
            raise self.error
        return "{}", 1


def _call(provider):
    service = AdvancedCommentService(provider=provider)
    return asyncio.run(service._llm_call_json_schema("system", "user", model="m", effort="low",
                                                      schema={}, schema_name="test"))


def test_failed_call_raises_alert_and_next_success_clears_it():
    with pytest.raises(openai.RateLimitError):
        _call(_Provider(_openai_error("insufficient_quota", "You exceeded your current quota")))
    assert [a["provider"] for a in provider_alerts.current()] == ["openai"]

    _call(_Provider())
    assert provider_alerts.current() == []
