"""Export descriptive freedom-pilot results and blind expert packets."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.research.runner import digest, read, verify_manifest, write_new
from scripts.benchmark_llm_freedom import MODES


def report(folder):
    manifest = verify_manifest(folder)
    records = []
    for c in read(folder / "schedule.json"):
        p = folder / "runs" / c["run_id"] / "result.json"
        r = read(p)
        if digest(r) != p.with_suffix(".sha256").read_text().strip():
            raise ValueError("Modified result")
        records.append(r)
    summary = []
    all_assertions = []
    for mode in MODES:
        rows = [r for r in records if r["variant_id"] == mode]
        assertions = [a for r in rows for a in r.get("review", {}).get("assertions", [])]
        counts = Counter(a["verdict"] for a in assertions)
        words = [len(re.sub(r"\[pv:[^]]*\]", "", s["comment"]).split()) for r in rows for s in r.get("samples", [])]
        generation = [c["seconds"] for r in rows for c in r.get("calls", []) if c["stage"] == "generate"]
        reviewed = [r for r in rows if r.get("review", {}).get("assertions")]
        row = {"mode": mode, "scheduled": len(rows), "generated_or_archived": sum(r["status"] == "completed" for r in rows),
               "generation_failures": sum(r["status"] != "completed" for r in rows),
               "mechanical_pass": sum(r.get("validation", {}).get("passed", False) for r in rows),
               "reviews_available": len(reviewed), "screen_pass": sum(r.get("screening_status") == "screen_pass" for r in rows),
               "comments_with_contradictions": sum(any(a["verdict"] == "contradicted" for a in r.get("review", {}).get("assertions", [])) for r in rows),
               "comments_with_unsupported": sum(any(a["verdict"] == "unsupported" for a in r.get("review", {}).get("assertions", [])) for r in rows),
               "comments_with_unassessable": sum(any(a["verdict"] == "unassessable" for a in r.get("review", {}).get("assertions", [])) for r in rows),
               "assertions": len(assertions), **{k: counts[k] for k in ("supported", "contradicted", "unsupported", "unassessable")},
               "mean_words": statistics.mean(words) if words else None,
               "mean_new_generation_seconds": statistics.mean(generation) if generation else None}
        summary.append(row)
        for r in rows:
            all_assertions.extend({"case": r["case_id"], "mode": mode, **a} for a in r.get("review", {}).get("assertions", []))
    write_new(folder / "summary.json", {"modes": summary, "human_ratings": "unavailable",
              "judge": "same composer-2.5 model; screening only", "calls": sum(len(r.get("calls", [])) for r in records),
              "usage_and_cost": "unavailable", "manifest_hash": digest(manifest)})
    with (folder / "metrics.csv").open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    write_new(folder / "assertion-reviews.json", all_assertions)
    lines = ["# LLM freedom: 20-position controlled pilot", "", "## Design", "",
             "One seeded, nonterminal, claim-bearing position from each of the 20 School games. Four arms per position (80 outputs). The template and bounded-editor outputs are archived paired outputs from the earlier benchmark; grounded rewriting and free-form commentary are fresh generations. All use the same stored position and engine continuation; no new engine searches or future recorded moves are provided.", "",
             "Grounded rewriting receives the rule-derived claims, their supporting positions, and engine evidence, and may freely paraphrase them with citations. Free-form commentary receives the board and engine analysis without the approved claim catalog and may infer plans. Thus this contrast changes both freedom and access to the claim catalog; it is not an isolated measurement of linguistic freedom alone. Both fresh modes target 45–85 words, with no truncation or length-based rerun.", "",
             "Generation and factual screening both use Cursor composer-2.5. The reviewer sees the actual text plus a common evidence catalog, without a mode label. This is same-model screening, not independent validation or a factual-accuracy score. Unsupported means not established by the supplied evidence; it does not necessarily mean false. Contradictions also require expert confirmation.", "",
             "## Results", "", "| Mode | Outputs | Mechanical pass | Reviews available | Screen pass | Comments flagged contradicted | Comments flagged unsupported | Mean words |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        words = f"{r['mean_words']:.1f}" if r['mean_words'] is not None else "N/A"
        lines.append(f"| {r['mode']} | {r['generated_or_archived']}/{r['scheduled']} | {r['mechanical_pass']}/{r['scheduled']} | {r['reviews_available']}/{r['scheduled']} | {r['screen_pass']}/{r['scheduled']} | {r['comments_with_contradictions']} | {r['comments_with_unsupported']} | {words} |")
    lines += ["", "Mechanical checks cover nonempty output, valid evidence IDs, and legality/alignment of explicitly tagged PVs. Grounded rewriting additionally requires a citation. Citations alone do not validate prose. Screen pass requires mechanical success and every reviewed assertion plus the overall review to be supported. Missing or malformed reviews never count as passes. A reviewer may still miss a claim.", "",
              "All raw text is retained, including flagged outputs. There is no hidden template replacement or regeneration of rejected prose. The expert packets contain the raw outputs, not just those that passed. Nothing in this experiment changes production commentary.", "", "## Paired examples", ""]
    # Choose three deterministic cases with new-mode flags so readers can inspect the failure modes.
    chosen = sorted({r["case_id"] for r in records if r["variant_id"] in ("grounded_rewrite", "free_form") and r.get("screening_status") != "screen_pass"})[:3]
    if not chosen: chosen = [c["id"] for c in manifest["cases"][:3]]
    for cid in chosen:
        lines += [f"### {cid}", ""]
        for mode in MODES:
            r = next(r for r in records if r["case_id"] == cid and r["variant_id"] == mode)
            lines += [f"**{mode}**", ""]
            for s in r.get("samples", []): lines += ["> " + s["comment"].replace("\n", " "), ""]
            flagged = [a for a in r.get("review", {}).get("assertions", []) if a["verdict"] != "supported"]
            for a in flagged[:3]: lines += [f"- Model screen: **{a['verdict']}** — {a['text']} Reason: {a['reason']}"]
            if r.get("error"): lines += ["- Generation failure: " + r["error"]]
            if r.get("review", {}).get("error"): lines += ["- Review unavailable: " + r["review"]["error"]]
            lines += [""]
    lines += ["## Interpretation", "",
              "Compare comment-level rates first; assertion counts differ with output length and are not independent samples. With only 20 positions, one generation per mode, and no human ratings, these results are exploratory. No significance test or claim of improved usefulness is warranted.", "",
              "The key thesis question is whether greater expressiveness improves explanations enough to offset evidence gaps. Novel but correct chess knowledge may be screened as unsupported; conversely, the same-model reviewer may accept a shared mistake. Expert review should distinguish these cases and check timing, side ownership, causal claims, and omitted key points.", "",
              "## Human review and reproducibility", "",
              "The evaluator has four counterbalanced rounds, with one output per position in each round. Each of two raters receives all 80 raw outputs independently. Keep evaluator/private_key.csv private. The catalog and fixed engine lines are available for research verification, while the public packet shows only the position, lines, and anonymous prose. Use the existing correctness/relevance/usefulness/concision rubric, add notes identifying questionable assertions, and record disagreements before adjudication.", "",
              "Artifacts: manifest.json, source.json, snapshots/, runs/, summary.json, metrics.csv, assertion-reviews.json, evaluator/public/. Generation/review prompts and raw responses are in each immutable result. API usage/cost is unavailable because the SDK does not report reliable token usage. Archived arms have no new generation timing; generation and review timings are kept separately in call records.", "",
              "Run: `python scripts/benchmark_llm_freedom.py --experiment data/research/experiments/freedom-new-run`, followed by `python scripts/report_llm_freedom.py --experiment data/research/experiments/freedom-new-run`. Use a new directory for new generations; --resume verifies and skips prior results."]
    (folder / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    from app.research.bank import ingest
    from app.research.evaluator import build_evaluator
    bank = ingest(folder, ROOT / "data/research/results.sqlite")
    packets = build_evaluator(folder, folder / "evaluator", seed=manifest["seed"])
    print(json.dumps({"summary": summary, "bank": bank, "packets": packets}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    report(parser.parse_args().experiment.resolve())
