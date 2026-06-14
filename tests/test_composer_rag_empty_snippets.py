"""Composer post-processing when MASTER ANNOTATIONS did not yield snippets."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.core.commentary import llm_call_log
from app.core.commentary.advanced_comment_service import AdvancedCommentService


class _FakeComposerProvider:
    name = "fake"

    def is_configured(self) -> bool:
        return True

    async def text_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        max_output_tokens: Any = None,
    ) -> tuple[str, int]:
        return "", 0

    async def json_schema_call(
        self,
        system: str,
        user: str,
        *,
        model: str,
        effort: str,
        schema: dict[str, Any],
        schema_name: str,
        max_output_tokens: Any = None,
    ) -> tuple[str, int]:
        payload = {
            "named_motifs": [],
            "text": "White keeps central tension while developing.",
            "better_alternative": "",
            "rag_idea_used": "claimed reuse of master prose",
            "rag_applied": True,
        }
        return json.dumps(payload), 12


def test_rag_applied_false_when_snippets_empty_writes_postcheck(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOG_LLM_TO_FILE", "1")
    gid = "rag-empty-snippets-game-id"
    gtok = llm_call_log.set_game_context(gid)
    try:

        async def _run() -> None:
            svc = AdvancedCommentService(provider=_FakeComposerProvider())
            dbg: dict[str, Any] = {
                "detail_level": "minimal",
                "game_digest": {"strategic_archetype": "other"},
                "rag_snippets_used": [],
                "motif_hint_keys": [],
                "composer_forcing_pv_bracket": None,
            }
            text = await svc.analyze_and_compose_raw_text(
                "===DYNAMIC===\nPLAYED_MOVE minimal stub",
                key_moment_type=None,
                move_category="positional",
                llm_debug=dbg,
            )
            assert text.strip()
            assert dbg.get("composer_rag_applied") is False
            assert dbg.get("composer_rag_idea_used") == ""

            path = tmp_path / "logs" / "llm" / f"{gid}.jsonl"
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            rows = [json.loads(line) for line in lines]
            post = [r for r in rows if r.get("pass_name") == "composer_postcheck"]
            assert post, "expected composer_postcheck row"
            assert post[-1]["postcheck"]["rag_applied"] is False
            assert post[-1]["postcheck"]["rag_idea_used"] == ""

        asyncio.run(_run())
    finally:
        llm_call_log.reset_game_context(gtok)
