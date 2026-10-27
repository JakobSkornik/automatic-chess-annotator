"""Paired controlled ablation of bounded editor context profiles.

All profiles see the same frozen B11 facts; only the user prompt's context
profile changes. This is a diagnostic prompt-component study, not a quality
claim or a held-out evaluation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from app.core.commentary.llm_providers import make_llm_provider
from app.core.commentary.phases.composer import _audience_view, render_facts_template
from app.core.commentary.phases.composer.framing import comment_archetype
from app.core.commentary.phases.composer.grounded import (
    EDITOR_VERSION,
    EDITOR_SCHEMA,
    EDITOR_SYSTEM,
    editor_prompt,
    render_editor_response,
)
from app.models.comment_facts import CommentFacts
from app.research import runner
from app.core.commentary.phases.composer.evidence import EVIDENCE_VERSION

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("claims_only", "board_grounded")
VARIANTS = ({"id": "T", "profile": "template"},
            {"id": "C", "profile": "claims_only"},
            {"id": "E", "profile": "board_grounded"})
PROTOCOL = "prompt-context-ablation-v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_snapshot() -> dict[str, str]:
    # Capture every app/script source plus requirements: facts, evidence,
    # provider adapters and rendering all affect what this experiment means.
    return runner.source_files()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _provider_controls(provider: str, model: str, effort: str, max_output_tokens: int) -> dict:
    import os
    if provider in ("cursor", "grok"):
        from app.core.commentary.llm_providers import CURSOR_EFFORT_PARAMS
        effort_effective = "passed" if model in CURSOR_EFFORT_PARAMS else "not_exposed_by_model"
        token_effective = "ignored_by_provider"
    else:
        # This provider applies effort only to models that use the reasoning API.
        from app.core.commentary.llm_providers import _openai_model_uses_reasoning_api
        effort_effective = "passed" if _openai_model_uses_reasoning_api(model) else "not_applicable_for_model"
        token_effective = "passed_as_max_output_tokens"
    return {"truncation_retries": int(os.environ.get("COMMENTARY_TRUNCATION_RETRIES", "1")),
            "requested": {"effort": effort, "max_output_tokens": max_output_tokens,
                           "temperature": "uncontrolled_not_exposed"},
            "effective": {"effort": effort_effective, "max_output_tokens": token_effective,
                          "temperature": "uncontrolled_not_exposed"}}


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def _case_snapshot(parent: Path, case: dict, manifest: dict, *, level: str) -> dict:
    from app.core.commentary.rules.facts_builder import build_comment_facts

    snap = runner.load_snapshot(parent, case["id"])
    if snap.get("case_id") != case["id"] or snap.get("dataset_hash") != manifest.get("dataset_hash"):
        raise ValueError(f"{case['id']}: snapshot does not match the parent case/dataset")
    row, event, probe = runner.replay_inputs(snap)
    b11 = next((v for v in manifest["variants"] if v["id"] == "B11"), None)
    if b11 is None or b11.get("horizon") != "adaptive_feature" or b11.get("claims") != "contrastive":
        raise ValueError("Parent experiment lacks the expected B11 selector configuration")
    facts = build_comment_facts(row, event, depth=manifest["engine"]["depth"], prober=probe,
                                horizon_policy=b11["horizon"], claim_policy=b11["claims"])
    if facts is None:
        raise ValueError(f"{case['id']}: facts builder abstained")
    audience_facts = _audience_view(facts, level)
    system = EDITOR_SYSTEM
    prompts = {profile: editor_prompt(audience_facts, level, context_profile=profile)
               for profile in PROFILES}
    template = render_facts_template(audience_facts, archetype=comment_archetype(None))
    return {"case_id": case["id"], "case": case, "parent_snapshot_hash": snap["snapshot_hash"],
            "facts": audience_facts.model_dump(mode="json"), "template": template,
            "system": system, "schema": EDITOR_SCHEMA,
            "prompts": prompts, "prompt_hashes": {k: digest({"system": system, "user": v,
                                                                "schema": EDITOR_SCHEMA}) for k, v in prompts.items()}}


def prepare(parent: Path, folder: Path, *, provider: str, model: str, effort: str = "low",
            max_output_tokens: int = 1500, replicates: int = 3, seed: int = 20261020) -> dict:
    """Rebuild all B11 facts, freeze identical paired inputs, and schedule T/C/E."""
    parent, folder = Path(parent).resolve(), Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(f"Experiment directory already exists: {folder}")
    if provider not in {"openai", "cursor", "grok"}:
        raise ValueError("provider must be openai, cursor, or grok")
    if not model or replicates < 1 or max_output_tokens < 1:
        raise ValueError("model, positive replicates and positive output limit are required")
    parent_manifest = runner.verify_manifest(parent)
    if parent_manifest.get("track") != "component":
        raise ValueError("Parent must be a component experiment with frozen engine snapshots")
    runner.configure(parent_manifest)
    cases = parent_manifest.get("cases") or []
    if not cases:
        raise ValueError("Parent experiment has no cases")
    level = parent_manifest.get("audience", "intermediate")
    # Build every case before creating output. A broken case aborts preparation
    # instead of silently changing the paired sample set.
    snapshots = [_case_snapshot(parent, case, parent_manifest, level=level) for case in cases]
    source = source_snapshot()
    schedule = [{"run_id": f"{folder.name}--{case['id']}-{v['id']}-r{rep}-a0",
                 "case_id": case["id"], "variant_id": v["id"], "replicate": rep}
                for case in cases for v in VARIANTS for rep in range(replicates)]
    random.Random(seed).shuffle(schedule)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {p: package_version(p) for p in
                                ("openai", "cursor-sdk", "python-chess", "pydantic")}}
    effective_model = model
    manifest = {"protocol": PROTOCOL, "experiment_id": folder.name,
        "dataset_status": "diagnostic_not_held_out", "track": "prompt_ablation",
        "parent_experiment_id": parent_manifest["experiment_id"],
        "parent_manifest_hash": runner.digest(parent_manifest),
        "parent_source_hash": parent_manifest["source_hash"],
        "dataset_hash": digest([s["case"] for s in snapshots]), "source_hash": digest(source),
        "provider": provider, "model_requested": model, "model_id_submitted": effective_model,
        "effective_model": {"status": "unverified_provider_alias", "value": None},
        "provider_controls": _provider_controls(provider, model, effort, max_output_tokens),
        "replicates": replicates, "seed": seed, "audience": level,
        "variants": list(VARIANTS), "cases": [s["case"] for s in snapshots],
        "rubric_version": "prompt-context-ablation-v1",
        "editor_version": EDITOR_VERSION,
        "evidence_version": EVIDENCE_VERSION,
        "schedule_hash": digest(schedule),
        "scope": "Same freshly recomputed B11 facts and editor contract; context profile is the only prompt treatment. Diagnostic comparison; no human quality claim.",
        "provider_call_semantics": "One logical call per C/E cell. OpenAI provider_trace records output caps, usage and token-truncation retries; SDK transport retries and Cursor internal retries are not enumerated. Logical calls are not HTTP-attempt counts.",
        "environment": environment, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    folder.mkdir(parents=True, exist_ok=False)
    _write_new(folder / "source.json", source)
    _write_new(folder / "manifest.json", manifest)
    (folder / "manifest.sha256").write_text(digest(manifest), encoding="utf-8")
    for snap in snapshots:
        snap["snapshot_hash"] = digest(snap)
        _write_new(folder / "snapshots" / f"{snap['case_id']}.json", snap)
    _write_new(folder / "schedule.json", schedule)
    return {"experiment_id": folder.name, "cases": len(cases), "scheduled_runs": len(schedule)}


def verify_manifest(folder: Path, *, require_current_source: bool = True) -> dict:
    folder = Path(folder).resolve()
    manifest = runner.read(folder / "manifest.json")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Not a prompt-ablation experiment")
    if digest(manifest) != (folder / "manifest.sha256").read_text(encoding="utf-8").strip():
        raise ValueError("Experiment manifest changed")
    if digest(runner.read(folder / "source.json")) != manifest["source_hash"]:
        raise ValueError("Source snapshot changed")
    if require_current_source and digest(source_snapshot()) != manifest["source_hash"]:
        raise ValueError("Current harness/composer source differs; prepare a new experiment")
    schedule = runner.read(folder / "schedule.json")
    if digest(schedule) != manifest.get("schedule_hash"):
        raise ValueError("Experiment schedule changed")
    return manifest


def _load_snapshot(folder: Path, case_id: str) -> dict:
    snap = runner.read(folder / "snapshots" / f"{case_id}.json")
    if digest({k: v for k, v in snap.items() if k != "snapshot_hash"}) != snap["snapshot_hash"]:
        raise ValueError(f"Snapshot hash mismatch: {case_id}")
    return snap


def _token_usage(raw_usage: Any) -> dict:
    if isinstance(raw_usage, int) and raw_usage > 0:
        return {"status": "reported", "tokens": raw_usage}
    return {"status": "unknown", "tokens": None}


def _run_cell(folder: Path, cell: dict, provider_obj: Any, *, timeout: float = 120) -> dict:
    import asyncio

    manifest = verify_manifest(folder)
    # New runs freeze the truncation policy as well as the initial output cap.
    import os
    retries = manifest["provider_controls"].get("truncation_retries")
    if retries is not None and retries != int(os.environ.get("COMMENTARY_TRUNCATION_RETRIES", "1")):
        raise ValueError("Truncation retry setting differs from the frozen experiment")
    snap = _load_snapshot(folder, cell["case_id"])
    variant = next(v for v in manifest["variants"] if v["id"] == cell["variant_id"])
    facts = CommentFacts(**snap["facts"])
    prompt_hash = snap["prompt_hashes"].get(variant["profile"])
    started = time.perf_counter()
    row = {**cell, "experiment_id": manifest["experiment_id"], "snapshot_hash": snap["snapshot_hash"],
           "prompt_hash": prompt_hash, "status": "completed", "model_status": "not_applicable",
           "fallback": False, "usage": {"status": "not_applicable", "tokens": None},
           "calls": [], "samples": []}
    output = snap["template"]
    if variant["profile"] != "template":
        profile = variant["profile"]
        call = {"provider": manifest["provider"], "model": manifest["model_id_submitted"],
                "system": snap["system"], "user": snap["prompts"][profile],
                "schema": snap["schema"], "schema_name": "grounded_comment"}
        row["calls"].append(call)
        call_started = time.perf_counter()
        try:
            raw, usage = asyncio.run(asyncio.wait_for(provider_obj.json_schema_call(
                call["system"], call["user"], model=call["model"], effort=manifest["provider_controls"]["requested"]["effort"],
                schema=call["schema"], schema_name=call["schema_name"],
                max_output_tokens=manifest["provider_controls"]["requested"]["max_output_tokens"]), timeout=timeout))
            call["raw_response"] = raw
            call["usage"] = _token_usage(usage)
            call["generation_seconds"] = time.perf_counter() - call_started
            row["usage"] = call["usage"]
            row["model_status"] = "response_received"
            output = render_editor_response(raw, facts, manifest["audience"], comment_archetype(None))
            row["actual_renderer"] = "bounded_llm"
        except Exception as exc:
            row["fallback"] = True
            row["model_status"] = "invalid_response" if "raw" in locals() else "failed"
            if "raw" not in locals():
                row["usage"] = _token_usage(getattr(exc, "token_usage", None))
                call["usage"] = row["usage"]
            call["error"] = f"{type(exc).__name__}: {exc}"
            call["generation_seconds"] = time.perf_counter() - call_started
            output = snap["template"]
            row["actual_renderer"] = "template_fallback"
        finally:
            call["provider_trace"] = getattr(provider_obj, "last_request", None)
    else:
        row["actual_renderer"] = "template"
    row["samples"] = [{"sample_id": digest(cell)[:24], "case_id": cell["case_id"],
                       "replicate": cell["replicate"], "comment": output,
                       "template": snap["template"], "facts": snap["facts"]}]
    row["generation_seconds"] = time.perf_counter() - started
    return row


def run(folder: Path, *, resume: bool = False, limit: int | None = None,
        provider_factory=make_llm_provider, timeout: float = 120) -> dict:
    folder = Path(folder).resolve()
    manifest = verify_manifest(folder)
    provider_obj = provider_factory(manifest["provider"])
    if not provider_obj.is_configured():
        raise ValueError(f"{manifest['provider']} provider is not configured")
    full_schedule = runner.read(folder / "schedule.json")
    schedule = full_schedule
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
    if resume:
        pending = []
        for cell in full_schedule:
            existing = folder / "runs" / cell["run_id"] / "result.json"
            if not existing.exists():
                pending.append(cell)
                continue
            saved = runner.read(existing)
            if digest(saved) != existing.with_suffix(".sha256").read_text(encoding="utf-8").strip():
                raise ValueError(f"Saved result changed: {cell['run_id']}")
        schedule = pending
    if limit is not None:
        schedule = schedule[:limit]
    outcomes = []
    for cell in schedule:
        dest = folder / "runs" / cell["run_id"] / "result.json"
        if dest.exists():
            saved = runner.read(dest)
            if digest(saved) != dest.with_suffix(".sha256").read_text(encoding="utf-8").strip():
                raise ValueError(f"Saved result changed: {cell['run_id']}")
            if not resume:
                raise FileExistsError("Run exists; use --resume to verify and skip")
            outcomes.append(saved)
            continue
        try:
            result = _run_cell(folder, cell, provider_obj, timeout=timeout)
        except Exception as exc:
            result = {**cell, "experiment_id": manifest["experiment_id"], "status": "failed",
                      "model_status": "failed", "fallback": False, "calls": [], "samples": [],
                      "error": f"{type(exc).__name__}: {exc}"}
        _write_new(dest, result)
        dest.with_suffix(".sha256").write_text(digest(result), encoding="utf-8")
        outcomes.append(result)
    return {"scheduled_total": len(full_schedule), "selected_this_run": len(schedule),
            "completed": sum(r["status"] == "completed" for r in outcomes),
            "failed": sum(r["status"] != "completed" for r in outcomes),
            "fallbacks": sum(bool(r.get("fallback")) for r in outcomes)}


def report(folder: Path) -> dict:
    folder = Path(folder).resolve()
    manifest = verify_manifest(folder, require_current_source=False)
    rows = []
    schedule = runner.read(folder / "schedule.json")
    for cell in schedule:
        path = folder / "runs" / cell["run_id"] / "result.json"
        if not path.exists():
            rows.append({**cell, "status": "pending"})
            continue
        row = runner.read(path)
        if digest(row) != path.with_suffix(".sha256").read_text(encoding="utf-8").strip():
            raise ValueError(f"Result hash mismatch: {cell['run_id']}")
        rows.append(row)
    metrics = {}
    for variant in manifest["variants"]:
        group = [r for r in rows if r["variant_id"] == variant["id"]]
        metrics[variant["id"]] = {"scheduled": len(group),
            "pending": sum(r["status"] == "pending" for r in group),
            "completed": sum(r["status"] == "completed" for r in group),
            "failed": sum(r["status"] == "failed" for r in group),
            "fallbacks": sum(bool(r.get("fallback")) for r in group),
            "model_responses": sum(r.get("model_status") == "response_received" for r in group),
            "usage_unknown": sum(r.get("usage", {}).get("status") == "unknown" for r in group)}
    summary = {"experiment_id": manifest["experiment_id"], "scope": manifest["scope"],
               "provider_call_semantics": manifest["provider_call_semantics"],
               "metrics": metrics, "paired_text_changed": {},
               "provider_controls": manifest["provider_controls"],
               "effective_model": manifest["effective_model"],
               "independent_quality": "not_measured",
               "quality_claim": "none; requires blinded human ratings"}
    for left, right in (("T", "C"), ("T", "E"), ("C", "E")):
        pairs = {}
        for r in rows:
            if r["status"] != "completed" or not r.get("samples"):
                continue
            pairs[(r["case_id"], r["replicate"], r["variant_id"])] = r["samples"][0]["comment"]
        shared = [(key, pairs[(key[0], key[1], left)], pairs[(key[0], key[1], right)])
                  for key in pairs if key[2] == left and (key[0], key[1], right) in pairs]
        summary["paired_text_changed"][f"{left}-{right}"] = {
            "paired": len(shared), "changed": sum(a != b for _, a, b in shared)}
    report_hash = digest(summary)
    reports = folder / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    immutable = reports / f"{report_hash}.json"
    if immutable.exists():
        if digest(runner.read(immutable)) != report_hash:
            raise ValueError("Content-addressed report snapshot changed")
        immutable_hash = immutable.with_suffix(".sha256")
        if not immutable_hash.exists() or immutable_hash.read_text(encoding="utf-8").strip() != report_hash:
            raise ValueError("Content-addressed report hash is missing or invalid")
    else:
        _write_new(immutable, summary)
        immutable.with_suffix(".sha256").write_text(report_hash, encoding="utf-8")

    dest = folder / "report.json"
    sidecar = folder / "report.json.sha256"
    if dest.exists():
        if not sidecar.exists() or digest(runner.read(dest)) != sidecar.read_text(encoding="utf-8").strip():
            raise ValueError("Existing latest report hash is missing or invalid")
    # Replace only our verified derived latest view. Content-addressed snapshots
    # above preserve every earlier state (pending, partial, or complete).
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=folder,
                                     prefix="report-", suffix=".tmp", delete=False) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        temp_path = Path(stream.name)
    os.replace(temp_path, dest)
    sidecar.write_text(report_hash, encoding="utf-8")
    return summary
