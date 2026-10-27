"""Frozen-position pilot of LLM freedom, with retained raw output and screening."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.research.runner import digest, read, write_new, source_files, verify_manifest

MODES = ["template", "bounded", "grounded_rewrite", "free_form"]
OUTPUT_SCHEMA = {"type": "object", "properties": {
    "text": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["text", "evidence_ids"], "additionalProperties": False}
COMMON = ("Write one useful chess annotation for an intermediate player, aiming for 45-85 words. "
          "Explain the move and its consequences. Distinguish the position now from a possible continuation. "
          "Return JSON text and evidence_ids. Do not use tools, files, browsing, or outside game history. "
          "If you include a variation, use [pv:SAN SAN ...] containing an exact supplied line. ")
GROUND = COMMON + ("Rewrite freely in your own prose, using ONLY the supplied evidence catalog. "
                   "You may reorganize and connect facts but must not invent reasons or claims. "
                   "List the IDs supporting your factual assertions in evidence_ids. "
                   "Feature changes do not by themselves prove causation. Do not expose IDs in the prose.")
FREE = COMMON + ("Generate your own commentary from the supplied board and engine analysis. "
                 "You may infer strategic ideas and explain plans; no approved claim list is supplied. "
                 "Do not invent future recorded moves. Leave evidence_ids empty.")


def prepare(folder, seed=20260907):
    import chess
    suite = read(ROOT / "data/experiments/school20-inputs-20260906/suite.json")
    old = ROOT / "data/experiments/school20-benchmark-20260906"
    llm = ROOT / "data/experiments/school20-llm-benchmark-20260906"
    originals = {r["case"]: r for r in read(old / "report.json")["runs"]}
    bounded = {r["case"]: r for r in read(llm / "report.json")["runs"]}
    cases, snapshots = [], []
    rng = random.Random(seed)
    for c in suite["cases"]:
        p = Path(originals[c["id"]]["artifact_dir"]) / "game.json"
        q = Path(bounded[c["id"]]["artifact_dir"]) / "game.json"
        game, edited = read(p), read(q)
        eligible = [i for i, m in enumerate(game["moves"]) if m.get("final_comment")
                    and (m.get("comment_facts") or {}).get("claims")
                    and not m["comment_facts"].get("terminal_result")
                    and ((edited["moves"][i].get("debug") or {}).get("renderings") or {}).get("intermediate") == "llm"]
        if not eligible:
            raise ValueError(f"No eligible archived paired annotation for {c['id']}")
        i = rng.choice(eligible)
        move, edit = game["moves"][i], edited["moves"][i]
        facts = move["comment_facts"]
        if facts != edit["comment_facts"]:
            raise ValueError("Archived evidence differs across renderer arms")
        fen = game["metadata"]["initial_fen"] if i == 0 else game["moves"][i-1]["fen"]
        board = chess.Board(fen)
        position = {"fen": fen, "turn": "White" if board.turn else "Black",
                    "pieces": {chess.square_name(s): piece.symbol() for s, piece in board.piece_map().items()},
                    "played_san": move["san"], "played_uci": move["uci"]}
        catalog = {"position": position,
                   "evaluation": {k: facts.get(k) for k in ("eval_cp", "eval_mate", "engine", "depth", "verdict")}}
        catalog["evaluation"]["perspective"] = "White; evaluation backed up from engine continuation"
        lines = []
        for name, branch in (("main", facts), ("alternative", facts.get("better_alternative") or {})):
            if branch.get("display_line"):
                catalog[name + ".line"] = branch["display_line"]
                lines.append(branch["display_line"])
            if name == "alternative" and branch:
                catalog["alternative.evaluation"] = {k: branch.get(k) for k in ("san", "uci", "eval_cp", "verdict", "is_inferior")}
            for j, claim in enumerate(branch.get("claims", [])):
                catalog[f"{name}.claim{j}"] = claim
        evidence = {"fen_before": fen, "move_san": move["san"], "catalog": catalog, "lines": lines}
        # The free arm has identical board/search evidence, but no rule-derived claim list.
        free_input = {"position": position, "evaluation": {k: catalog["evaluation"][k] for k in ("eval_cp", "eval_mate", "engine", "depth", "perspective")},
                      "lines": [{"san": line["san"], "fens": line["fens"]} for line in lines],
                      "alternative": {k: (facts.get("better_alternative") or {}).get(k) for k in ("san", "uci", "eval_cp")}}
        cid = f"{c['id']}-p{i+1}"
        case = {"id": cid, "source_group": c["id"], "family_group": c["id"], "split": "diagnostic",
                "category": "freedom_pilot", "initial_fen": fen, "target_ply": i+1,
                "input_hash": digest(evidence), "fen_before": fen, "move_san": move["san"],
                "provenance": c["provenance"]}
        snap = {"case_id": cid, "evidence": evidence, "free_input": free_input, "facts": facts,
                "archived_texts": {"template": move["comment"], "bounded": edit["comment"]},
                "source_artifacts": {str(p): hashlib.sha256(p.read_bytes()).hexdigest(), str(q): hashlib.sha256(q.read_bytes()).hexdigest()}}
        snap["snapshot_hash"] = digest(snap)
        cases.append(case); snapshots.append(snap)
    sources = source_files()
    manifest = {"experiment_id": folder.name, "protocol": "llm-freedom-pilot-v1", "track": "renderer",
                "dataset_status": "diagnostic_not_held_out", "dataset_hash": digest(cases), "source_hash": digest(sources),
                "cases": cases, "variants": [{"id": m, "renderer": m} for m in MODES], "seed": seed,
                "model": "composer-2.5", "provider": "cursor", "replicates": 1, "rubric_version": "freedom-v1",
                "generation_prompts": {"grounded_rewrite": GROUND, "free_form": FREE}, "output_schema": OUTPUT_SCHEMA,
                "selection": "One seeded random nonterminal, claim-bearing paired annotation per game; archived template and bounded arms reused",
                "screening": "Same-model assertion screening; not independent human validation; raw rejected text retained",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "source.json", sources); write_new(folder / "manifest.json", manifest)
    (folder / "manifest.sha256").write_text(digest(manifest))
    for snap in snapshots:
        write_new(folder / "snapshots" / f"{snap['case_id']}.json", snap)
    schedule = [{"run_id": f"{folder.name}--{c['id']}-{mode}-r0-a0", "case_id": c["id"], "variant_id": mode} for c in cases for mode in MODES]
    rng.shuffle(schedule); write_new(folder / "schedule.json", schedule)


async def cell(folder, cell):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from app.core.commentary.llm_providers import CursorProvider
    from app.research.freedom_validation import validate_output, REVIEW_SCHEMA, REVIEW_SYSTEM, parse_review
    manifest = verify_manifest(folder)
    if digest(source_files()) != manifest["source_hash"]:
        raise ValueError("Source changed: prepare a new experiment")
    snap = read(folder / "snapshots" / f"{cell['case_id']}.json")
    if snap["snapshot_hash"] != digest({k: v for k, v in snap.items() if k != "snapshot_hash"}):
        raise ValueError("Snapshot changed")
    dest = folder / "runs" / cell["run_id"] / "result.json"
    if dest.exists():
        if digest(read(dest)) != dest.with_suffix(".sha256").read_text().strip():
            raise ValueError("Result changed")
        return
    provider = CursorProvider(); mode = cell["variant_id"]; started = time.perf_counter()
    result = {**cell, "experiment_id": folder.name, "replicate": 0, "attempt": 0,
              "snapshot_hash": snap["snapshot_hash"], "status": "failed", "samples": [], "calls": [],
              "requested_renderer": mode, "actual_renderer": mode, "cost_basis": {"status": "unavailable"}}
    async def call(stage, system, user, schema):
        t = time.perf_counter(); record = {"stage": stage, "system": system, "user": user, "schema": schema}
        result["calls"].append(record)
        try:
            raw, usage = await provider.json_schema_call(system, json.dumps(user, ensure_ascii=False), model=manifest["model"],
                        effort="low", schema=schema, schema_name="freedom_"+stage, max_output_tokens=2500)
            record.update(raw=raw, reported_usage=usage, seconds=time.perf_counter()-t)
            return raw
        except Exception as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter()-t)
            raise
    try:
        if mode in snap["archived_texts"]:
            output = {"text": snap["archived_texts"][mode], "evidence_ids": []}
            result["generation_origin"] = "archived"
        else:
            user = snap["evidence"] if mode == "grounded_rewrite" else snap["free_input"]
            raw = await call("generate", manifest["generation_prompts"][mode], user, OUTPUT_SCHEMA)
            output = json.loads(raw); result["generation_origin"] = "fresh"
        result["raw_output"] = output
        validation = validate_output(output, snap["evidence"], mode == "grounded_rewrite")
        result["validation"] = validation
        if not isinstance(output.get("text"), str) or not output["text"].strip():
            raise ValueError("Missing generated text")
        # Review all four arms with the same evidence and no arm label.
        try:
            review_raw = await call("review", REVIEW_SYSTEM + " Use only exact catalog keys as evidence_ids. Do not use tools, files, or external sources.", {"evidence": snap["evidence"], "text": output["text"]}, REVIEW_SCHEMA)
            result["review"] = parse_review(review_raw, snap["evidence"])
        except Exception as exc:
            result["review"] = {"semantic_pass": False, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        case = next(c for c in manifest["cases"] if c["id"] == cell["case_id"])
        def ui_line(line):
            return {**line, "line_san": line["san"]} if line else None
        facts = snap["facts"]
        ui_facts = {"display_line": ui_line(facts.get("display_line")),
                    "better_alternative": {"display_line": ui_line((facts.get("better_alternative") or {}).get("display_line"))}}
        result["samples"] = [{"sample_id": digest(cell)[:24], "target_ply": case["target_ply"],
                              "fen_before": case["fen_before"], "move_san": case["move_san"],
                              "comment": output["text"], "facts": ui_facts}]
        result.update(status="completed", explanation_outcome="delivered",
                      screening_status="screen_pass" if validation["passed"] and result["review"].get("semantic_pass") else "screen_flagged")
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", error_category=type(exc).__name__, explanation_outcome="failed")
    result["seconds"] = time.perf_counter()-started
    write_new(dest, result); dest.with_suffix(".sha256").write_text(digest(result))


def run(folder):
    schedule = read(folder / "schedule.json")
    started = time.perf_counter()
    def worker(c):
        log = folder / "worker-logs" / f"{c['run_id']}.log"; log.parent.mkdir(exist_ok=True)
        try:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--experiment", str(folder), "--cell", c["run_id"]],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=360)
            log.write_text(p.stdout+p.stderr, encoding="utf-8")
            if p.returncode:
                raise RuntimeError(f"Worker exit {p.returncode}")
        except Exception as exc:
            dest = folder / "runs" / c["run_id"] / "result.json"
            if not dest.exists():
                record = {**c, "experiment_id": folder.name, "status": "failed", "error": str(exc), "samples": []}
                write_new(dest, record); dest.with_suffix(".sha256").write_text(digest(record))
        r = read(folder / "runs" / c["run_id"] / "result.json")
        print(json.dumps({"case": c["case_id"], "mode": c["variant_id"], "status": r["status"], "screen": r.get("screening_status")}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, schedule))
    write_new(folder / f"execution-{time.time_ns()}.json", {"wall_seconds": time.perf_counter()-started, "workers": 4})


def main():
    import asyncio
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True); p.add_argument("--cell"); p.add_argument("--resume", action="store_true")
    args = p.parse_args(); folder = args.experiment.resolve()
    if args.cell:
        c = next(c for c in read(folder / "schedule.json") if c["run_id"] == args.cell)
        asyncio.run(cell(folder, c)); return
    if not folder.exists(): prepare(folder)
    elif not args.resume: raise FileExistsError("Use --resume to preserve and verify saved results")
    run(folder)


if __name__ == "__main__":
    main()
