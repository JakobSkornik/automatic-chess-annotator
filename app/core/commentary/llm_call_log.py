"""Opt-in JSONL logging of LLM prompts and responses per game (LOG_LLM_TO_FILE)."""

from __future__ import annotations

import json
import logging
import os
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GAME_ID: ContextVar[str | None] = ContextVar("llm_call_log_game_id", default=None)
_PLY: ContextVar[int | None] = ContextVar("llm_call_log_ply", default=None)
_EPISODE_INDEX: ContextVar[int | None] = ContextVar(
    "llm_call_log_episode_index", default=None
)
_PASS_LABEL: ContextVar[str | None] = ContextVar(
    "llm_call_log_pass_label", default=None
)

_lock_registry_lock = threading.Lock()
_game_locks: dict[str, threading.Lock] = {}
_game_seq: dict[str, int] = {}


def is_enabled() -> bool:
    return os.environ.get("LOG_LLM_TO_FILE", "").strip().lower() in ("1", "true")


def set_game_context(game_id: str | None) -> Any:
    """Return token for ``reset_game_context``."""
    return _GAME_ID.set(game_id)


def reset_game_context(token: Any) -> None:
    _GAME_ID.reset(token)


def set_move_context(
    *,
    ply: int | None = None,
    episode_index: int | None = None,
    pass_label: str | None = None,
) -> tuple[Any, Any, Any]:
    """Return tokens tuple for ``reset_move_context``."""
    return (
        _PLY.set(ply),
        _EPISODE_INDEX.set(episode_index),
        _PASS_LABEL.set(pass_label),
    )


def reset_move_context(tokens: tuple[Any, Any, Any]) -> None:
    t0, t1, t2 = tokens
    _PASS_LABEL.reset(t2)
    _EPISODE_INDEX.reset(t1)
    _PLY.reset(t0)


def _sanitized_game_id(game_id: str) -> str:
    """Avoid path traversal or illegal filenames."""
    return game_id.replace("/", "_").replace("\\", "_").replace("..", "_")


def _path_for_game(game_id: str) -> Path:
    return Path("logs") / "llm" / f"{_sanitized_game_id(game_id)}.jsonl"


def _lock_for_game(game_id: str) -> threading.Lock:
    with _lock_registry_lock:
        if game_id not in _game_locks:
            _game_locks[game_id] = threading.Lock()
        return _game_locks[game_id]


def log_call(
    *,
    pass_name: str,
    system: str,
    user: str,
    response: str,
    model: str,
    effort: str | None,
    schema_name: str | None,
    token_usage: int | None,
    elapsed_ms: float,
    ok: bool,
    error: str | None = None,
) -> int | None:
    if not is_enabled():
        return None
    gid = _GAME_ID.get()
    if not gid:
        return None

    safe_gid = _sanitized_game_id(gid)
    lk = _lock_for_game(safe_gid)
    with lk:
        seq = _game_seq.get(safe_gid, 0) + 1
        _game_seq[safe_gid] = seq

        dt = datetime.now(timezone.utc)
        ms = dt.microsecond // 1000
        ts = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"

        record: dict[str, Any] = {
            "seq": seq,
            "ts": ts,
            "game_id": gid,
            "pass_name": pass_name,
            "model": model,
            "effort": effort,
            "schema_name": schema_name,
            "token_usage": token_usage,
            "elapsed_ms": round(elapsed_ms, 3),
            "ok": ok,
            "error": error,
            "prompt_chars": len(system) + len(user),
            "response_chars": len(response),
            "prompt_token_est": (len(system) + len(user)) // 4,
            "system": system,
            "user": user,
            "response": response,
        }
        ply_v = _PLY.get()
        if ply_v is not None:
            record["ply"] = ply_v
        ep_v = _EPISODE_INDEX.get()
        if ep_v is not None:
            record["episode_index"] = ep_v
        pl_v = _PASS_LABEL.get()
        if pl_v is not None:
            record["pass_label"] = pl_v
        path = _path_for_game(gid)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning("llm_call_log: failed to write %s: %s", path, e)
        return seq


def append_postcheck(*, ref_seq: int, payload: dict[str, Any]) -> None:
    """Append a correction row after composer post-processing (e.g. rag_applied overlap fix)."""
    if not is_enabled():
        return
    gid = _GAME_ID.get()
    if not gid:
        return
    safe_gid = _sanitized_game_id(gid)
    lk = _lock_for_game(safe_gid)
    with lk:
        dt = datetime.now(timezone.utc)
        ms = dt.microsecond // 1000
        ts = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"
        record: dict[str, Any] = {
            "seq": ref_seq,
            "ts": ts,
            "game_id": gid,
            "pass_name": "composer_postcheck",
            "ref_seq": ref_seq,
            "postcheck": payload,
        }
        ply_v = _PLY.get()
        if ply_v is not None:
            record["ply"] = ply_v
        ep_v = _EPISODE_INDEX.get()
        if ep_v is not None:
            record["episode_index"] = ep_v
        pl_v = _PASS_LABEL.get()
        if pl_v is not None:
            record["pass_label"] = pl_v
        path = _path_for_game(gid)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning(
                "llm_call_log append_postcheck: failed to write %s: %s", path, e
            )
