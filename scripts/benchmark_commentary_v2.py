"""School development benchmark: selector factorial + separate sentence-plan trial.

Run from the repository with --stage all. Existing artifacts are verified and
reused only with --resume; raw provider failures and fallbacks remain visible.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import concurrent.futures
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.research import runner


def prepare(folder: Path):
    previous = runner.verify_manifest(ROOT / "data/research/experiments/llm-freedom-pilot-20260907")
    targets = {c["source_group"]: c["target_ply"] for c in previous["cases"]}
    suite = runner.read(ROOT / "data/experiments/school20-inputs-20260906/suite.json")
    suite["version"] = "school20-commentary-v2-development"
    suite["selection"] = "Reuse the 20 previously inspected freedom-pilot targets; not held out"
    for case in suite["cases"]:
        case["required_comment_plies"] = [targets[case["id"]]]
    path = folder.parent / (folder.name + "-inputs") / "suite.json"
    if path.exists():
        if runner.read(path) != suite:
            raise ValueError("Prepared input suite changed")
    else:
        runner.write_new(path, suite)
    return runner.prepare(path, folder, depth=16, probe_depth=6, extra_targets=False)


def records(folder: Path):
    result = []
    for cell in runner.read(folder / "schedule.json"):
        path = folder / "runs" / cell["run_id"] / "result.json"
        row = runner.read(path)
        if runner.digest(row) != path.with_suffix(".sha256").read_text().strip():
            raise ValueError("Run artifact changed")
        result.append(row)
    return result


def prepare_renderer(parent: Path):
    from app.core.commentary.phases.composer.sentence_plan import VERSION, PROMPT_VERSION, SYSTEM, SCHEMA
    manifest = runner.verify_manifest(parent)
    folder = parent / "renderer"
    if folder.exists():
        runner.verify_manifest(folder)
        return folder
    source_rows = {r["case_id"]: r for r in records(parent) if r["variant_id"] == "B11"}
    if any(source_rows[c["id"]]["status"] != "completed" for c in manifest["cases"]):
        raise ValueError("Cannot prepare renderer trial with missing B11 facts")
    new = {**manifest, "experiment_id": parent.name + "-renderer", "track": "renderer",
           "protocol": "controlled-sentence-plan-v1", "model": "composer-2.5", "provider": "cursor",
           "variants": [{"id": "T", "renderer": "template"}, {"id": "S", "renderer": VERSION}],
           "renderer_version": VERSION, "prompt_version": PROMPT_VERSION, "system": SYSTEM, "schema": SCHEMA,
           "scope": "Identical B11 facts; legal replay and approved sentence realization, not independent rule-truth verification",
           "parent_manifest_hash": runner.digest(manifest)}
    runner.write_new(folder / "manifest.json", new)
    (folder / "manifest.sha256").write_text(runner.digest(new))
    runner.write_new(folder / "source.json", runner.read(parent / "source.json"))
    for case in new["cases"]:
        row = source_rows[case["id"]]
        snapshot = {"case_id": case["id"], "facts": row["samples"][0]["facts"],
                    "source_run_hash": runner.digest(row), "source_snapshot_hash": row["snapshot_hash"]}
        snapshot["snapshot_hash"] = runner.digest(snapshot)
        runner.write_new(folder / "snapshots" / f"{case['id']}.json", snapshot)
    schedule = [{"run_id": f"{new['experiment_id']}--{c['id']}-{mode}-r0-a0", "case_id": c["id"], "variant_id": mode}
                for c in new["cases"] for mode in ("T", "S")]
    runner.write_new(folder / "schedule.json", schedule)
    return folder


async def renderer_cell(folder: Path, cell: dict):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from app.models.comment_facts import CommentFacts
    from app.core.commentary.phases.composer import _audience_view, render_facts_template
    from app.core.commentary.phases.composer.sentence_plan import catalog, default_plan, has_editorial_choice, render_plan, planner_prompt
    from app.core.commentary.llm_providers import CursorProvider
    manifest = runner.verify_manifest(folder)
    if runner.digest(runner.source_files()) != manifest["source_hash"]:
        raise ValueError("Source changed; create a new experiment")
    dest = folder / "runs" / cell["run_id"] / "result.json"
    if dest.exists():
        if runner.digest(runner.read(dest)) != dest.with_suffix(".sha256").read_text().strip():
            raise ValueError("Existing renderer result changed")
        return
    snapshot = runner.load_snapshot(folder, cell["case_id"])
    case = next(c for c in manifest["cases"] if c["id"] == cell["case_id"])
    facts = _audience_view(CommentFacts(**snapshot["facts"]), manifest["audience"])
    baseline = render_facts_template(facts)
    result = {**cell, "experiment_id": manifest["experiment_id"], "replicate": 0, "attempt": 0,
              "snapshot_hash": snapshot["snapshot_hash"], "requested_renderer": cell["variant_id"],
              "actual_renderer": "template", "calls": [], "samples": [], "status": "running",
              "cost_basis": {"status": "unavailable"}}
    started = time.perf_counter()
    text = baseline
    if cell["variant_id"] == "S":
        try:
            evidence = catalog(facts)
            plan = json.dumps(default_plan(evidence))
            if has_editorial_choice(evidence):
                provider = CursorProvider()
                if not provider.is_configured():
                    raise ValueError("Cursor provider is not configured")
                prompt_input = planner_prompt(facts, manifest["audience"])
                call = {"system": manifest["system"], "user": prompt_input, "schema": manifest["schema"]}
                result["calls"].append(call)
                call_started = time.perf_counter()
                try:
                    plan, usage = await asyncio.wait_for(provider.json_schema_call(
                        manifest["system"], json.dumps(prompt_input), model=manifest["model"], effort="low",
                        schema=manifest["schema"], schema_name="sentence_plan", max_output_tokens=1500), timeout=150)
                    call.update(raw=plan, reported_usage=usage)
                except Exception as exc:
                    call["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    call["seconds"] = time.perf_counter() - call_started
            else:
                result["call_skipped"] = "no_editorial_choice"
            result["raw_plan"] = plan
            rendered = render_plan(plan, facts)
            text = rendered["text"]
            result.update(actual_renderer="sentence_plan_llm" if result["calls"] else "sentence_plan_deterministic",
                          verification=rendered)
        except Exception as exc:
            result.update(fallback_reason=f"{type(exc).__name__}: {exc}",
                          error_category=type(exc).__name__)
    sample = {"sample_id": runner.digest(cell)[:24], "target_ply": case["target_ply"],
              "fen_before": case["fen_before"], "move_san": facts.san, "comment": text,
              "facts": facts.model_dump(mode="json")}
    sample["audit"] = runner.audit_sample(sample, case)
    result.update(status="completed", explanation_outcome="delivered", samples=[sample],
                  changed_from_template=text != baseline, seconds=time.perf_counter() - started)
    runner.write_new(dest, result)
    dest.with_suffix(".sha256").write_text(runner.digest(result))


def run_renderer(folder: Path):
    cells = runner.read(folder / "schedule.json")
    def work(cell):
        dest = folder / "runs" / cell["run_id"] / "result.json"
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--experiment", str(folder),
                                     "--stage", "cell", "--cell", cell["run_id"]], cwd=ROOT,
                                     capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=210)
            log = folder / "worker-logs" / (cell["run_id"] + ".log")
            log.parent.mkdir(exist_ok=True)
            log.write_text(process.stdout + process.stderr, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"Worker exited {process.returncode}; see {log.name}")
        except Exception as exc:
            if not dest.exists():
                row = {**cell, "experiment_id": runner.verify_manifest(folder)["experiment_id"],
                       "status": "failed", "samples": [], "error": str(exc)}
                runner.write_new(dest, row)
                dest.with_suffix(".sha256").write_text(runner.digest(row))
        row = runner.read(dest)
        print(json.dumps({"case": cell["case_id"], "variant": cell["variant_id"],
                          "status": row["status"], "renderer": row.get("actual_renderer"),
                          "fallback": row.get("fallback_reason")}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work, cells))


def report(folder: Path):
    from app.research.bank import ingest
    from app.research.evaluator import build_evaluator
    manifest = runner.verify_manifest(folder)
    all_rows = records(folder)
    metrics = []
    for variant in manifest["variants"]:
        rows = [r for r in all_rows if r["variant_id"] == variant["id"]]
        samples = [s for r in rows for s in r.get("samples", [])]
        lengths = [len((s["facts"].get("display_line") or {}).get("line_uci", [])) for s in samples]
        unknown = sum(c.get("probe_status") == "unknown" for s in samples
                      for c in s["facts"].get("selection_metadata", {}).get("horizon", {}).get("candidates", []))
        metrics.append({"variant": variant["id"], "scheduled": len(rows), "completed": len(samples),
                        "audit_issues": sum(len(s.get("audit", {}).get("issues", [])) for s in samples),
                        "mean_display_plies": statistics.mean(lengths) if lengths else None,
                        "unknown_probe_candidates": unknown})
    paired = []
    for case in manifest["cases"]:
        by_mode = {r["variant_id"]: r["samples"][0] for r in all_rows if r["case_id"] == case["id"] and r.get("samples")}
        if "B00" in by_mode and "B11" in by_mode:
            a, b = by_mode["B00"], by_mode["B11"]
            paired.append({"case": case["id"], "changed_text": a["comment"] != b["comment"],
                           "baseline": a["comment"], "combined": b["comment"],
                           "selection": b["facts"].get("selection_metadata", {})})
    renderer = records(folder / "renderer") if (folder / "renderer/schedule.json").exists() else []
    new_rows = [r for r in renderer if r["variant_id"] == "S"]
    renderer_metrics = {"scheduled": len(new_rows), "completed": sum(r["status"] == "completed" for r in new_rows),
                        "actual_renderers": dict(Counter(r.get("actual_renderer", "failed") for r in new_rows)),
                        "calls": sum(len(r.get("calls", [])) for r in new_rows),
                        "fallbacks": sum(bool(r.get("fallback_reason")) for r in new_rows),
                        "changed": sum(r.get("changed_from_template", False) for r in new_rows),
                        "audit_issues": sum(len(s["audit"]["issues"]) for r in renderer for s in r.get("samples", []))}
    summary = {"selector_metrics": metrics, "paired_changed": sum(x["changed_text"] for x in paired),
               "paired_positions": len(paired), "renderer_metrics": renderer_metrics,
               "human_ratings": "unavailable", "scope": "development component benchmark; no quality claim"}
    runner.write_new(folder / "summary-v2.json", summary)
    runner.write_new(folder / "paired-examples-v2.json", paired)
    lines = ["# Commentary improvements: development benchmark", "", "## Scope", "",
             "Twenty previously inspected positions, one per School game. This is development data, not held out. Fresh depth-16 engine snapshots with depth-6 probes, one thread, identical evidence across four selector arms. This is forced-target component evaluation, not full-game annotation coverage.", "",
             "B00: current horizon/current claims. B10: refined adaptive horizon/current claims. B01: current horizon/refined contrastive claims. B11: both refinements. All use the repaired deterministic template.", "", "## Mechanical results", ""]
    for m in metrics:
        length = f"{m['mean_display_plies']:.2f}" if m['mean_display_plies'] is not None else "unavailable"
        lines.append(f"- {m['variant']}: {m['completed']}/{m['scheduled']} outputs, {m['audit_issues']} audit issues; mean displayed line {length} plies.")
    lines += ["", f"Combined-selector text changed in {summary['paired_changed']}/{len(paired)} paired positions. Changed wording and shorter lines do not establish better explanations.", "",
              "## Separate controlled-rewriting trial", "",
              f"Template and sentence-plan rendering use identical B11 facts. {renderer_metrics['completed']}/{renderer_metrics['scheduled']} sentence-plan outputs delivered; {renderer_metrics['calls']} logical provider calls; {renderer_metrics['fallbacks']} fallbacks; {renderer_metrics['changed']} texts changed. Actual renderers: {renderer_metrics['actual_renderers']}. Mechanical audit issues across both renderer arms: {renderer_metrics['audit_issues']}.", "",
              "The model selects approved clause forms and sentence groups. Chess assertions, side ownership, time anchors, evaluations and PVs are application-owned. Legal replay and faithful realization are checked; source-rule semantic correctness is NOT independently certified. This is controlled rewriting, not unrestricted prose. Raw responses and rejected plans are retained. Token usage/cost is unavailable unless independently established; SDK zeros are not proof of free usage.", "",
              "## Examples", "", "First three case IDs with changed combined-selector text, chosen to illustrate changes rather than estimate typical quality.", ""]
    for p in [p for p in paired if p["changed_text"]][:3]:
        lines += [f"### {p['case']}", "", "Baseline:", "> " + p["baseline"], "", "Combined:", "> " + p["combined"], ""]
    lines += ["## Interpretation and next step", "",
              "The benchmark establishes execution, evidence consistency and the extent of output changes. It does not establish higher correctness, usefulness or educational benefit. Guid and the second rater must independently assess anonymous outputs, endpoint adequacy and concept relevance. Public evaluator packets contain all delivered outputs, including fallback templates; private keys and raw implementation metadata stay separate.", "",
              "Existing pilot results are unchanged. New snapshots and sources are frozen for this run. The report should be read with docs/commentary-research-2026-10-08.md and the selector version metadata.", "",
              "Reproduce with `python scripts/benchmark_commentary_v2.py --experiment data/research/experiments/NEW-ID --stage all`. Use a new ID for changed sources; `--resume` verifies and reuses existing run artifacts, including recorded failures. No automatic cherry-picked regeneration."]
    (folder / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for target in [folder, *([folder / "renderer"] if renderer else [])]:
        ingest(target, ROOT / "data/research/results.sqlite")
        build_evaluator(target, target / "evaluator", seed=manifest["seed"])
    print(json.dumps(summary), flush=True)
    return summary


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "prepare", "selectors", "rewrite", "report", "cell"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cell")
    args = parser.parse_args()
    folder = args.experiment.resolve()
    if args.stage == "cell":
        cell = next(c for c in runner.read(folder / "schedule.json") if c["run_id"] == args.cell)
        asyncio.run(renderer_cell(folder, cell))
        return
    if args.stage in ("prepare", "all"):
        if folder.exists() and not args.resume:
            raise FileExistsError("Choose a new directory or --resume")
        if not folder.exists():
            print(json.dumps(prepare(folder)), flush=True)
    if args.stage in ("selectors", "all"):
        runner.snapshot(folder)
        print(json.dumps(runner.run(folder, resume=args.resume)), flush=True)
        audit = runner.audit(folder)
        if audit["issues"]:
            raise ValueError(f"Selector audit failed: {audit}")
    if args.stage in ("rewrite", "all"):
        run_renderer(prepare_renderer(folder))
    if args.stage in ("report", "all"):
        summary = report(folder)
        if any(m["completed"] != m["scheduled"] or m["audit_issues"] for m in summary["selector_metrics"]):
            raise SystemExit(1)
        if (summary["renderer_metrics"]["fallbacks"] or summary["renderer_metrics"]["audit_issues"]
                or summary["renderer_metrics"]["completed"] != summary["renderer_metrics"]["scheduled"]):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
