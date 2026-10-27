import asyncio
import json
from unittest.mock import AsyncMock

from app.research import runner
from scripts.benchmark_commentary_v2 import prepare, renderer_cell, records
from app.core.commentary.phases.composer.sentence_plan import SYSTEM, SCHEMA, catalog, default_plan
from tests.test_sentence_plan import facts


def test_school_targets_have_only_history_through_target(tmp_path):
    from app.evaluation import parse_game
    folder = tmp_path / "school"
    prepared = prepare(folder)
    manifest = runner.verify_manifest(folder)
    assert prepared["cases"] == 20
    assert len({c["source_group"] for c in manifest["cases"]}) == 20
    for case in manifest["cases"]:
        game = parse_game(case["pgn"])
        assert len(list(game.mainline_moves())) == case["target_ply"]
        assert not any(n.comment for n in game.mainline())
        assert game.headers["Result"] == "*"


def setup_renderer(folder):
    f = facts()
    f.claims.append(f.claims[0].model_copy(update={"text": "The pawn occupies e4."}))
    case = {"id": "case", "target_ply": 1, "fen_before": f.display_line.start_fen}
    source = runner.source_files()
    manifest = {"experiment_id": "trial", "source_hash": runner.digest(source), "audience": "intermediate",
                "cases": [case], "model": "unused", "system": SYSTEM, "schema": SCHEMA}
    runner.write_new(folder / "source.json", source)
    runner.write_new(folder / "manifest.json", manifest)
    (folder / "manifest.sha256").write_text(runner.digest(manifest))
    snapshot = {"case_id": "case", "facts": f.model_dump(mode="json")}
    snapshot["snapshot_hash"] = runner.digest(snapshot)
    runner.write_new(folder / "snapshots/case.json", snapshot)
    cell = {"run_id": "trial-case-S-r0-a0", "case_id": "case", "variant_id": "S"}
    runner.write_new(folder / "schedule.json", [cell])
    return f, cell


def test_rejected_model_text_is_retained_and_fallback_is_explicit(tmp_path, monkeypatch):
    _, cell = setup_renderer(tmp_path)
    raw = json.dumps({"text": "Black is winning."})
    provider = type("Provider", (), {"is_configured": lambda self: True,
        "json_schema_call": AsyncMock(return_value=(raw, 0))})()
    monkeypatch.setattr("app.core.commentary.llm_providers.CursorProvider", lambda: provider)
    asyncio.run(renderer_cell(tmp_path, cell))
    row = records(tmp_path)[0]
    assert row["actual_renderer"] == "template"
    assert row["fallback_reason"]
    assert row["raw_plan"] == raw
    assert "Black is winning" not in row["samples"][0]["comment"]


def test_accepted_plan_preserves_identical_facts_and_resumes_without_call(tmp_path, monkeypatch):
    f, cell = setup_renderer(tmp_path)
    raw = json.dumps(default_plan(catalog(f)))
    provider = type("Provider", (), {"is_configured": lambda self: True,
        "json_schema_call": AsyncMock(return_value=(raw, 0))})()
    monkeypatch.setattr("app.core.commentary.llm_providers.CursorProvider", lambda: provider)
    asyncio.run(renderer_cell(tmp_path, cell))
    asyncio.run(renderer_cell(tmp_path, cell))
    row = records(tmp_path)[0]
    assert row["actual_renderer"] == "sentence_plan_llm"
    assert row["samples"][0]["facts"] == f.model_dump(mode="json")
    assert provider.json_schema_call.await_count == 1
