"""Tests for opt-in JSONL logging of LLM calls (LOG_LLM_TO_FILE)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.commentary import llm_call_log


def test_log_call_disabled_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOG_LLM_TO_FILE", raising=False)
    tok = llm_call_log.set_game_context("should-not-write")
    try:
        llm_call_log.log_call(
            pass_name="x",
            system="s",
            user="u",
            response="r",
            model="m",
            effort="low",
            schema_name=None,
            token_usage=None,
            elapsed_ms=1.0,
            ok=True,
        )
    finally:
        llm_call_log.reset_game_context(tok)
    assert not (tmp_path / "logs" / "llm").exists()


def test_enabled_without_game_context_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LLM_TO_FILE", "1")
    llm_call_log.log_call(
        pass_name="x",
        system="s",
        user="u",
        response="r",
        model="m",
        effort="low",
        schema_name=None,
        token_usage=None,
        elapsed_ms=1.0,
        ok=True,
    )
    assert not (tmp_path / "logs").exists()


def test_enabled_with_game_two_calls_ordered_seq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LLM_TO_FILE", "true")
    gid = "game-seq-test-unique"
    tok = llm_call_log.set_game_context(gid)
    try:
        llm_call_log.log_call(
            pass_name="first",
            system="s1",
            user="u1",
            response="r1",
            model="m1",
            effort="medium",
            schema_name="game_digest",
            token_usage=10,
            elapsed_ms=12.3,
            ok=True,
        )
        llm_call_log.log_call(
            pass_name="second",
            system="s2",
            user="u2",
            response="r2",
            model="m2",
            effort=None,
            schema_name=None,
            token_usage=None,
            elapsed_ms=4.0,
            ok=False,
            error="SomethingError('x')",
        )
    finally:
        llm_call_log.reset_game_context(tok)

    path = tmp_path / "logs" / "llm" / f"{gid}.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    r1 = json.loads(lines[0])
    r2 = json.loads(lines[1])
    assert r1["seq"] == 1 and r2["seq"] == 2
    assert r1["game_id"] == gid and r2["game_id"] == gid
    assert r1["pass_name"] == "first" and r1["response"] == "r1"
    assert r1["prompt_chars"] == len("s1") + len("u1")
    assert r1["response_chars"] == len("r1")
    assert r1["prompt_token_est"] == r1["prompt_chars"] // 4
    assert r2["ok"] is False and r2["error"] == "SomethingError('x')"


def test_ply_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LLM_TO_FILE", "1")
    gid = "game-ply-test-unique"
    gtok = llm_call_log.set_game_context(gid)
    mtok = llm_call_log.set_move_context(ply=15)
    try:
        llm_call_log.log_call(
            pass_name="x",
            system="s",
            user="u",
            response="r",
            model="m",
            effort="low",
            schema_name=None,
            token_usage=None,
            elapsed_ms=1.0,
            ok=True,
        )
    finally:
        llm_call_log.reset_move_context(mtok)
        llm_call_log.reset_game_context(gtok)

    line = json.loads((tmp_path / "logs" / "llm" / f"{gid}.jsonl").read_text(encoding="utf-8"))
    assert line["ply"] == 15


def test_append_postcheck_writes_second_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LLM_TO_FILE", "1")
    gid = "game-postcheck-test"
    gtok = llm_call_log.set_game_context(gid)
    mtok = llm_call_log.set_move_context(ply=7, pass_label="key_moment")
    try:
        seq = llm_call_log.log_call(
            pass_name="composer_single",
            system="s",
            user="u",
            response='{"named_motifs":[],"text":"x","better_alternative":"","rag_idea_used":"","rag_applied":true}',
            model="m",
            effort="low",
            schema_name="chess_commentary_composer",
            token_usage=3,
            elapsed_ms=2.0,
            ok=True,
        )
        assert seq == 1
        llm_call_log.append_postcheck(
            ref_seq=seq,
            payload={"rag_applied": False, "rag_idea_used": "", "named_motifs": []},
        )
    finally:
        llm_call_log.reset_move_context(mtok)
        llm_call_log.reset_game_context(gtok)

    lines = (tmp_path / "logs" / "llm" / f"{gid}.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[1])
    assert row["pass_name"] == "composer_postcheck"
    assert row["ref_seq"] == 1
    assert row["postcheck"]["rag_applied"] is False


def test_file_isolation_per_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LLM_TO_FILE", "1")
    gid = "isolation-game"
    tok = llm_call_log.set_game_context(gid)
    try:
        llm_call_log.log_call(
            pass_name="only",
            system="s",
            user="u",
            response="r",
            model="m",
            effort="low",
            schema_name=None,
            token_usage=None,
            elapsed_ms=1.0,
            ok=True,
        )
    finally:
        llm_call_log.reset_game_context(tok)

    assert Path.cwd().resolve() == tmp_path.resolve()
    assert (tmp_path / "logs" / "llm" / f"{gid}.jsonl").is_file()
