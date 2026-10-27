from __future__ import annotations

import json

from app.models.comment_facts import Claim, CommentFacts
from app.research import prompt_ablation as ablation
from app.core.commentary.phases.composer.grounded import editor_prompt


def _facts() -> CommentFacts:
    return CommentFacts(
        ply=1, san="Nf3", uci="g1f3", mover="White", phase="mid",
        verdict="holds the balance", eval_cp=0,
        claims=[Claim(rule_id="piece_activity", text="White improves piece activity.",
                      text_state="White's pieces are more active.", beneficiary="white",
                      realization="immediate", support_ply=1,
                      support_fen="after", baseline_fen="before")],
    )


def _experiment(tmp_path, facts=None):
    facts = facts or _facts()
    source = ablation.source_snapshot()
    prompts = {profile: editor_prompt(facts, "intermediate", context_profile=profile)
               for profile in ablation.PROFILES}
    prompt_hashes = {profile: ablation.digest({"system": "same system", "user": prompt,
                                                "schema": ablation.EDITOR_SCHEMA})
                     for profile, prompt in prompts.items()}
    schedule = [{"run_id": f"r-{mode}", "case_id": "c1", "variant_id": mode, "replicate": 0}
                for mode in ("T", "C", "E")]
    manifest = {"protocol": ablation.PROTOCOL, "experiment_id": "test-ablation",
        "source_hash": ablation.digest(source), "scope": "test", "audience": "intermediate",
        "provider": "cursor", "model_id_submitted": "composer-2.5",
        "effective_model": {"status": "unverified_provider_alias", "value": None},
        "provider_call_semantics": "logical calls only",
        "provider_controls": {"requested": {"effort": "low", "max_output_tokens": 100}},
        "variants": list(ablation.VARIANTS), "schedule_hash": ablation.digest(schedule)}
    snap = {"case_id": "c1", "facts": facts.model_dump(mode="json"),
        "template": "Nf3 holds the balance.", "system": "same system", "schema": ablation.EDITOR_SCHEMA,
        "prompts": prompts, "prompt_hashes": prompt_hashes}
    snap["snapshot_hash"] = ablation.digest(snap)
    folder = tmp_path / "experiment"
    folder.mkdir()
    (folder / "source.json").write_text(json.dumps(source), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "manifest.sha256").write_text(ablation.digest(manifest), encoding="utf-8")
    (folder / "schedule.json").write_text(json.dumps(schedule), encoding="utf-8")
    (folder / "snapshots").mkdir()
    (folder / "snapshots" / "c1.json").write_text(json.dumps(snap), encoding="utf-8")
    return folder, snap


class _Provider:
    def __init__(self, raw, usage=0):
        self.raw, self.usage, self.calls = raw, usage, []

    def is_configured(self):
        return True

    async def json_schema_call(self, system, user, **kwargs):
        self.calls.append((system, user, kwargs))
        return self.raw, self.usage


def test_profiles_use_frozen_prompts_and_zero_usage_is_unknown(tmp_path):
    folder, snap = _experiment(tmp_path)
    assert snap["prompts"]["claims_only"] != snap["prompts"]["board_grounded"]
    provider = _Provider(json.dumps({"main": [{"id": 0, "form": "change"}], "alternative": []}))
    cell = {"run_id": "r", "case_id": "c1", "variant_id": "C", "replicate": 0}
    result = ablation._run_cell(folder, cell, provider)
    assert result["status"] == "completed"
    assert result["actual_renderer"] == "bounded_llm"
    assert result["prompt_hash"] == snap["prompt_hashes"]["claims_only"]
    assert provider.calls[0][1] == snap["prompts"]["claims_only"]
    assert provider.calls[0][2]["schema_name"] == "grounded_comment"
    assert result["usage"] == {"status": "unknown", "tokens": None}


def test_invalid_editor_response_is_recorded_as_template_fallback(tmp_path):
    folder, _ = _experiment(tmp_path)
    provider = _Provider('{"prose":"unapproved"}', usage=23)
    cell = {"run_id": "r", "case_id": "c1", "variant_id": "E", "replicate": 0}
    result = ablation._run_cell(folder, cell, provider)
    assert result["status"] == "completed"
    assert result["model_status"] == "invalid_response"
    assert result["fallback"] is True
    assert result["actual_renderer"] == "template_fallback"
    assert result["samples"][0]["comment"] == result["samples"][0]["template"]
    assert result["usage"] == {"status": "reported", "tokens": 23}


def test_board_grounded_uses_same_schema_name_as_claims_only(tmp_path):
    folder, snap = _experiment(tmp_path)
    provider = _Provider(json.dumps({"main": [{"id": 0, "form": "state"}], "alternative": []}))
    cell = {"run_id": "r-E", "case_id": "c1", "variant_id": "E", "replicate": 0}
    result = ablation._run_cell(folder, cell, provider)
    assert result["prompt_hash"] == snap["prompt_hashes"]["board_grounded"]
    assert provider.calls[0][2]["schema_name"] == "grounded_comment"


def test_provider_controls_disclose_cursor_ignored_limits():
    controls = ablation._provider_controls("cursor", "composer-2.5", "low", 1500)
    assert controls["requested"]["effort"] == "low"
    assert controls["effective"]["effort"] == "not_exposed_by_model"
    assert controls["effective"]["max_output_tokens"] == "ignored_by_provider"
    assert controls["effective"]["temperature"] == "uncontrolled_not_exposed"


def test_current_models_expose_effort_and_retry_policy(monkeypatch):
    monkeypatch.setenv("COMMENTARY_TRUNCATION_RETRIES", "0")
    for provider, model in (("openai", "gpt-6-luna"), ("grok", "grok-4.6")):
        controls = ablation._provider_controls(provider, model, "high", 32768)
        assert controls["effective"]["effort"] == "passed"
        assert controls["truncation_retries"] == 0


def test_frozen_retry_policy_rejects_changed_environment(tmp_path, monkeypatch):
    import pytest
    folder, _ = _experiment(tmp_path)
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provider_controls"]["truncation_retries"] = 0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "manifest.sha256").write_text(ablation.digest(manifest), encoding="utf-8")
    monkeypatch.setenv("COMMENTARY_TRUNCATION_RETRIES", "1")
    provider = _Provider("")
    with pytest.raises(ValueError, match="Truncation retry setting"):
        ablation._run_cell(folder, {"case_id": "c1", "variant_id": "C"}, provider)
    assert not provider.calls


def test_exhausted_truncation_keeps_usage_and_trace(tmp_path):
    from app.core.commentary.llm_runtime import OutputTruncatedError
    folder, _ = _experiment(tmp_path)
    class TruncatedProvider(_Provider):
        last_request = {"attempts": [{"max_output_tokens": 100, "truncated": True}], "token_usage": 123}
        async def json_schema_call(self, *args, **kwargs):
            raise OutputTruncatedError("openai", 123)
    result = ablation._run_cell(folder, {"run_id": "r-C", "case_id": "c1", "variant_id": "C", "replicate": 0}, TruncatedProvider(""))
    assert result["actual_renderer"] == "template_fallback"
    assert result["usage"] == {"status": "reported", "tokens": 123}
    assert result["calls"][0]["provider_trace"] == TruncatedProvider.last_request


def test_manifest_verifies_schedule_and_frozen_source(tmp_path):
    folder, _ = _experiment(tmp_path)
    ablation.verify_manifest(folder)
    (folder / "schedule.json").write_text("[]", encoding="utf-8")
    try:
        ablation.verify_manifest(folder)
    except ValueError as exc:
        assert "schedule" in str(exc)
    else:
        raise AssertionError("tampered schedule was accepted")


def test_manifest_source_snapshot_includes_evidence_module(tmp_path):
    folder, _ = _experiment(tmp_path)
    source_path = folder / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    key = "app/core/commentary/phases/composer/evidence.py"
    assert key in source
    source[key] += "\n# altered for test"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    try:
        ablation.verify_manifest(folder)
    except ValueError as exc:
        assert "source" in str(exc).lower()
    else:
        raise AssertionError("tampered source snapshot was accepted")


def test_prepare_schedules_every_parent_case_with_paired_replicates(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    source = {"fixture": "source"}
    manifest = {"protocol": "thesis-component-v1", "experiment_id": "parent",
        "track": "component", "source_hash": ablation.digest(source),
        "dataset_hash": "dataset-hash", "audience": "intermediate",
        "engine": {"depth": 16, "base_plies": 5, "max_plies": 12},
        "variants": [{"id": "B11", "horizon": "adaptive_feature", "claims": "contrastive"}],
        "cases": [{"id": "case-a"}, {"id": "case-b"}]}
    (parent / "source.json").write_text(json.dumps(source), encoding="utf-8")
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (parent / "manifest.sha256").write_text(ablation.digest(manifest), encoding="utf-8")
    seen = []

    def make_snapshot(parent_dir, case, parent_manifest, *, level):
        seen.append(case["id"])
        return {"case_id": case["id"], "case": case, "parent_snapshot_hash": "fresh",
            "facts": _facts().model_dump(mode="json"), "template": "template",
            "system": "system", "schema": ablation.EDITOR_SCHEMA,
            "prompts": {"claims_only": "C", "board_grounded": "E"},
            "prompt_hashes": {"claims_only": "hc", "board_grounded": "he"}}

    monkeypatch.setattr(ablation, "_case_snapshot", make_snapshot)
    output = tmp_path / "new-experiment"
    result = ablation.prepare(parent, output, provider="cursor", model="composer-2.5", replicates=2)
    assert result["cases"] == 2
    assert result["scheduled_runs"] == 12
    assert set(seen) == {"case-a", "case-b"}
    schedule = json.loads((output / "schedule.json").read_text(encoding="utf-8"))
    assert {row["case_id"] for row in schedule} == {"case-a", "case-b"}
    assert {row["variant_id"] for row in schedule} == {"T", "C", "E"}
    assert ablation.verify_manifest(output)["schedule_hash"] == ablation.digest(schedule)


def test_resume_limit_advances_through_full_schedule_and_reports_progress(tmp_path):
    folder, _ = _experiment(tmp_path)
    provider = _Provider(json.dumps({"main": [{"id": 0, "form": "change"}], "alternative": []}))
    factory = lambda _name: provider
    first = ablation.run(folder, limit=1, provider_factory=factory)
    assert first["scheduled_total"] == 3
    assert first["selected_this_run"] == 1
    pending = ablation.report(folder)
    assert sum(m["pending"] for m in pending["metrics"].values()) == 2
    second = ablation.run(folder, resume=True, limit=1, provider_factory=factory)
    assert second["selected_this_run"] == 1
    partial = ablation.report(folder)
    assert sum(m["completed"] for m in partial["metrics"].values()) == 2
    assert sum(m["pending"] for m in partial["metrics"].values()) == 1
    third = ablation.run(folder, resume=True, provider_factory=factory)
    assert third["selected_this_run"] == 1
    complete = ablation.report(folder)
    assert sum(m["completed"] for m in complete["metrics"].values()) == 3
    assert sum(m["pending"] for m in complete["metrics"].values()) == 0
    snapshots = list((folder / "reports").glob("*.json"))
    assert len(snapshots) == 3
    assert (folder / "report.json").exists()
    assert len(set(provider.calls[i][2]["schema_name"] for i in range(len(provider.calls)))) == 1


def test_provider_error_and_timeout_are_separate_template_fallbacks(tmp_path):
    import asyncio

    folder, _ = _experiment(tmp_path)
    cell = {"run_id": "r-C", "case_id": "c1", "variant_id": "C", "replicate": 0}

    class BrokenProvider(_Provider):
        async def json_schema_call(self, system, user, **kwargs):
            raise RuntimeError("provider unavailable")

    failed = ablation._run_cell(folder, cell, BrokenProvider(""))
    assert failed["status"] == "completed"
    assert failed["fallback"] is True
    assert failed["model_status"] == "failed"
    assert "provider unavailable" in failed["calls"][0]["error"]
    assert failed["usage"] == {"status": "unknown", "tokens": None}

    class SlowProvider(_Provider):
        async def json_schema_call(self, system, user, **kwargs):
            await asyncio.sleep(0.05)
            return self.raw, self.usage

    timed_out = ablation._run_cell(folder, cell, SlowProvider(""), timeout=0.001)
    assert timed_out["fallback"] is True
    assert timed_out["model_status"] == "failed"
    assert "TimeoutError" in timed_out["calls"][0]["error"]
