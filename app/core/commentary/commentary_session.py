"""Per-game, multi-turn LLM conversations for chess commentary."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.core.commentary.json_utils import strip_json_fence
from app.core.commentary.llm_runtime import output_token_budget, call_with_truncation_retry
from app.core.commentary.llm_providers import (
    _strip_sentinel_for_openai,
    _usage_total_cursor,
    cursor_model_selection,
    _usage_total_openai,
    _openai_model_uses_reasoning_api,
)

logger = logging.getLogger(__name__)


class CursorRunFailed(RuntimeError):
    """A Cursor agent run finished with status 'error' (no further detail from the SDK)."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Cursor commentary run failed: {run_id}")
        self.run_id = run_id


class CommentarySession:
    """A serialized provider conversation scoped to one game.

    Provider state is committed only after ``record_annotation`` confirms that
    the resulting annotation passed the application's contract checks.
    """

    def __init__(self, provider: Any, game_id: str) -> None:
        self.provider = provider
        self.game_id = str(game_id)
        self.is_commentary_session = True
        self._lock = asyncio.Lock()
        self._config: tuple[Any, ...] | None = None
        self._system: str | None = None
        self._model: str | None = None
        self._schema: dict[str, Any] | None = None
        self._schema_name: str | None = None
        self._max_output_tokens: int | None = None
        self._response_id: str | None = None
        self._cursor_agent: Any = None
        self._canonical_records: list[dict[str, Any]] = []
        self._delivered_record_count = 0
        self._pending_user: str | None = None
        self._pending_raw = False
        self._closed = False
        self.turn_count = 0
        self.last_request: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return str(getattr(self.provider, "name", "unknown"))

    @property
    def policy_key(self) -> str:
        return str(getattr(self.provider, "policy_key", self.name))

    @property
    def session_id(self) -> str:
        """Stable, non-secret identifier suitable for logs."""
        return self.game_id

    def logging_context(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "turn_count": self.turn_count,
                "provider": self.name}

    def is_configured(self) -> bool:
        return bool(self.provider.is_configured())

    def _bind_config(
        self, system: str, *, model: str, schema: dict[str, Any],
        schema_name: str, max_output_tokens: int | None,
    ) -> None:
        # Freeze the request contract for this conversation. Effort can be
        # provider/model specific per call, but schema and instructions cannot.
        fingerprint = (system, model, json.dumps(schema, sort_keys=True, ensure_ascii=False),
                       schema_name, max_output_tokens)
        if self._config is None:
            self._config = fingerprint
            self._system, self._model = system, model
            self._schema, self._schema_name = schema, schema_name
            self._max_output_tokens = max_output_tokens
        elif fingerprint != self._config:
            raise ValueError("Commentary session system, model, and schema are immutable")

    def _feedback_context(self, start: int | None = None) -> str:
        start = self._delivered_record_count if start is None else start
        records = self._canonical_records[start:]
        if not records:
            return ""
        context_records = [{key: value for key, value in record.items() if key != "request"}
                           for record in records]
        return (
            "\n\nEarlier commentary records from this game, in processing order, for context only. "
            "Their claims are not proof; use current move evidence as authoritative:\n"
            + json.dumps(context_records, ensure_ascii=False)
        )

    def _canonical_transcript(self) -> str:
        if not self._canonical_records:
            return ""
        rows = [
            rec
            for rec in self._canonical_records
        ]
        return ("Earlier commentary records from this game, in processing order, for context only. "
                "Their claims are not proof; use current move evidence as authoritative:\n"
                + json.dumps(rows, ensure_ascii=False))

    def _fresh_user(self, user: str) -> str:
        transcript = self._canonical_transcript()
        if transcript:
            return f"{transcript}\n\nCurrent move request:\n{user}"
        return user

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
        async with self._lock:
            if self._closed:
                raise RuntimeError("Commentary session is closed")
            if self._pending_raw:
                raise RuntimeError("Record the previous annotation before requesting another")
            self._bind_config(system, model=model, schema=schema,
                              schema_name=schema_name,
                              max_output_tokens=max_output_tokens)
            try:
                raw, usage = await self._call_provider(user, effort=effort)
            except BaseException:
                # A failed native call may have advanced provider-side state.
                # Discard it; the next call is reconstructed from committed
                # annotations only. Mutating calls are never blindly retried.
                await self._reset_native()
                self._pending_user = None
                self._pending_raw = False
                raise
            self._pending_user = user
            self._pending_raw = True
            self.turn_count += 1
            self._delivered_record_count = len(self._canonical_records)
            return raw, usage

    async def _call_provider(self, user: str, *, effort: str) -> tuple[str, int]:
        if self.name == "openai":
            return await self._openai_call(user, effort=effort)
        if self.name == "cursor":
            return await self._cursor_call(user, effort=effort)
        raise ValueError(f"Unsupported commentary provider: {self.name}")

    async def _openai_call(self, user: str, *, effort: str) -> tuple[str, int]:
        current = _strip_sentinel_for_openai(user)
        if self._response_id is None:
            context = self._fresh_user(current)
            input_value: Any = [
                {"role": "system", "content": [{"type": "input_text", "text": self._system}]},
                {"role": "user", "content": [{"type": "input_text", "text": context}]},
            ]
        else:
            prior = self._feedback_context()
            text = f"{prior}\n\n{current}" if prior else current
            input_value = [{"role": "user", "content": [{"type": "input_text", "text": text}]}]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_value,
            "text": {"format": {"type": "json_schema", "name": self._schema_name,
                                  "strict": True, "schema": self._schema}},
        }
        if self._response_id is not None:
            kwargs["previous_response_id"] = self._response_id
        if _openai_model_uses_reasoning_api(str(self._model)):
            kwargs["reasoning"] = {"effort": effort}
        budget = output_token_budget(str(self._model), effort, self._max_output_tokens)
        if budget is not None:
            kwargs["max_output_tokens"] = budget
        self.last_request = {"provider": self.name}
        response, usage = await call_with_truncation_retry(
            self.provider._client.responses.create, kwargs, provider=self.name,
            usage_total=_usage_total_openai, trace=self.last_request)
        response_id = getattr(response, "id", None)
        if not response_id:
            raise RuntimeError("OpenAI response did not include an id for conversation chaining")
        self._response_id = str(response_id)
        raw = getattr(response, "output_text", None) or ""
        if not raw and getattr(response, "output", None):
            try:
                raw = response.output[0].content[0].text
            except Exception:
                raw = ""
        return str(raw).strip(), usage

    def _cursor_instruction(self) -> str:
        schema_hint = (
            f"Return ONLY a JSON object matching schema '{self._schema_name}'. "
            "No markdown fences, no prose outside JSON.\n"
            + json.dumps(self._schema, ensure_ascii=False)
        )
        return f"{self._system.strip()}\n\n{schema_hint}".strip()

    def _create_cursor_agent(self, effort: str | None = None) -> Any:
        # CursorProvider imports lazily and keeps these handles in the provider
        # module so tests and existing provider wiring can patch them.
        from app.core.commentary import llm_providers as providers
        if providers._Agent is None:
            raise RuntimeError("Cursor SDK is not initialized")
        options = self.provider._agent_options(str(self._model), effort)
        return providers._Agent.create(options)

    async def _cursor_call(self, user: str, *, effort: str | None = None) -> tuple[str, int]:
        """Retry one failed Cursor run on a fresh agent rebuilt from committed records.

        A failed run may have left a draft in the agent's history, so the agent is
        closed first; the retry never sees anything but accepted annotations.
        """
        failed: list[str] = []
        for attempt in range(2):
            try:
                raw, usage = await self._cursor_attempt(user, effort=effort)
            except CursorRunFailed as exc:
                failed.append(exc.run_id)
                logger.warning("Cursor run %s failed (attempt %d/2)", exc.run_id, attempt + 1)
                await self._reset_native()
                if attempt == 1:
                    raise
                continue
            self.last_request["failed_runs"] = failed
            return raw, usage
        raise AssertionError("unreachable")

    async def _cursor_attempt(self, user: str, *, effort: str | None = None) -> tuple[str, int]:
        created_now = self._cursor_agent is None
        if created_now:
            create_task = asyncio.create_task(asyncio.to_thread(self._create_cursor_agent, effort))
            try:
                self._cursor_agent = await asyncio.shield(create_task)
            except asyncio.CancelledError:
                try:
                    orphan = await create_task
                    await asyncio.to_thread(orphan.close)
                except BaseException:
                    pass
                self._cursor_agent = None
                raise
            prefix = self._cursor_instruction()
            transcript = self._canonical_transcript()
            if transcript:
                prefix += "\n\n" + transcript
        else:
            prefix = ""
        turn = user if created_now else user + self._feedback_context()
        prompt = f"{prefix}\n\n{turn}".strip() if prefix else turn.strip()
        self.last_request = {
            "provider": self.name,
            "model_selection": cursor_model_selection(str(self._model), effort),
            "prompt": prompt,
            "agent_id": getattr(self._cursor_agent, "agent_id",
                                 getattr(self._cursor_agent, "id", None)),
        }

        def _run() -> Any:
            run = self._cursor_agent.send(prompt)
            return run.wait()

        run_task = asyncio.create_task(asyncio.to_thread(_run))
        try:
            result = await asyncio.shield(run_task)
        except asyncio.CancelledError:
            try:
                await run_task
            except BaseException:
                pass
            await self._reset_native()
            raise
        if getattr(result, "status", None) == "error":
            raise CursorRunFailed(str(getattr(result, "id", "?")))
        raw = getattr(result, "result", None) or ""
        return strip_json_fence(str(raw)), _usage_total_cursor(result)

    async def record_annotation(self, record: dict[str, Any]) -> None:
        """Commit verified annotation context after the app finishes a move."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Commentary session is closed")
            canonical = {key: record.get(key) for key in
                         ("ply", "san", "uci", "fen_after", "text", "rendering",
                          "contract_ok", "request")}
            if self._pending_raw:
                accepted_llm = (bool(canonical.get("contract_ok"))
                                and canonical.get("rendering") == "llm")
                if not accepted_llm:
                    # Native state contains an untrusted draft. Force a rebuild
                    # from committed annotations on the next request.
                    await self._reset_native()
                if accepted_llm and canonical.get("text"):
                    canonical["request"] = self._pending_user
                    self._canonical_records.append(canonical)
            elif canonical.get("contract_ok") and canonical.get("text"):
                # Moves that did not invoke an LLM (for example a template
                # annotation) can still be useful context for the next turn.
                self._canonical_records.append(canonical)
            self._pending_user = None
            self._pending_raw = False

    async def _reset_native(self) -> None:
        self._response_id = None
        self._delivered_record_count = 0
        agent, self._cursor_agent = self._cursor_agent, None
        if agent is not None:
            try:
                await asyncio.to_thread(agent.close)
            except Exception:
                logger.exception("Failed to close reset Cursor commentary agent")

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            await self._reset_native()
            self._closed = True

