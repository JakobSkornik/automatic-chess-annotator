import asyncio
import json
from unittest.mock import AsyncMock

from app.research.runner import read, verify_manifest
from scripts.benchmark_llm_freedom import prepare, cell


def test_selection_freezes_twenty_paired_positions_without_future(tmp_path):
    folder = tmp_path / "pilot"
    prepare(folder)
    manifest = verify_manifest(folder)
    assert len(manifest["cases"]) == 20
    assert len({c["source_group"] for c in manifest["cases"]}) == 20
    assert len(read(folder / "schedule.json")) == 80
    for c in manifest["cases"]:
        snap = read(folder / "snapshots" / f"{c['id']}.json")
        assert "claims" not in json.dumps(snap["free_input"])
        assert "archived_texts" not in snap["evidence"]
        assert set(snap["archived_texts"]) == {"template", "bounded"}
        assert snap["evidence"]["fen_before"] == c["fen_before"]


def test_flagged_prose_is_retained_without_template_fallback(tmp_path, monkeypatch):
    folder = tmp_path / "pilot"
    prepare(folder)
    review = {"assertions": [{"text": "Invented claim", "verdict": "unsupported", "evidence_ids": [], "reason": "Absent from evidence"}], "overall": "unsupported"}
    provider = type("Provider", (), {"json_schema_call": AsyncMock(side_effect=[
        (json.dumps({"text": "Invented claim", "evidence_ids": ["not-a-source"]}), 0),
        (json.dumps(review), 0)])})()
    monkeypatch.setattr("app.core.commentary.llm_providers.CursorProvider", lambda: provider)
    selected = next(c for c in read(folder / "schedule.json") if c["variant_id"] == "grounded_rewrite")
    asyncio.run(cell(folder, selected))
    result = read(folder / "runs" / selected["run_id"] / "result.json")
    assert result["status"] == "completed"
    assert result["screening_status"] == "screen_flagged"
    assert not result["validation"]["passed"]
    assert result["samples"][0]["comment"] == "Invented claim"
    assert result["actual_renderer"] == "grounded_rewrite"
    assert not result["review"]["semantic_pass"]
