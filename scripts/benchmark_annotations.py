"""Versioned API benchmark. See docs/annotation-benchmark.md for the protocol."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_suite(path: Path, case_ids: list[str] | None) -> dict:
    from app.evaluation import parse_game
    suite = json.loads(path.read_text(encoding="utf-8"))
    suite.pop("sha256", None)
    if not suite.get("version") or not suite.get("cases"):
        raise ValueError("Suite requires version and nonempty cases")
    seen = set()
    for case in suite["cases"]:
        cid = case["id"]
        if not re.fullmatch(r"[a-z0-9_-]+", cid) or cid in seen:
            raise ValueError("Case IDs must be unique safe filenames")
        seen.add(cid)
        game = parse_game(case["pgn"])
        plies = len(list(game.mainline_moves()))
        if any(type(p) is not int or not 1 <= p <= plies for p in case["required_comment_plies"]):
            raise ValueError(f"Required ply outside mainline: {cid}")
        # Strip comments, NAGs and variations from the input to avoid leakage.
        import chess.pgn
        clean = chess.pgn.Game()
        clean.headers = game.headers.copy()
        node = clean
        for move in game.mainline_moves():
            node = node.add_variation(move)
        case["pgn"] = clean.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
    if case_ids:
        if set(case_ids) - seen:
            raise ValueError("Unknown case requested")
        suite["cases"] = [c for c in suite["cases"] if c["id"] in case_ids]
    suite["sha256"] = sha(json.dumps(suite, sort_keys=True).encode())
    return suite


async def worker(request_path: Path):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    folder = request_path.parent
    # The SDK bridge also inherits the process cwd, independently of local.cwd.
    # Match normal backend startup; route only log files into the artifact folder.
    os.chdir(ROOT)
    os.environ["LOG_LLM_DIR"] = str(folder / "logs" / "llm")
    os.environ["HUMANIZATION_LEVEL"] = "0" if request["mode"] == "template" else "2"
    os.environ["LOG_LLM_TO_FILE"] = "1"
    os.environ["LOG_LLM_PROMPTS"] = "0"
    import httpx
    from fastapi import FastAPI
    from app.routes.jobs import router
    from app.core.worker import analysis_worker
    from app.core.engine.engine_connector import get_global_engine_connector
    from app.core.commentary.llm_providers import make_llm_provider
    from app.core.commentary.llm_policy import resolve_model
    if request["mode"] == "llm" and not make_llm_provider(request["provider"]).is_configured():
        raise ValueError("Requested LLM provider is not configured")
    engine = get_global_engine_connector()
    engine.engine.configure({"Threads": 1, "Clear Hash": None})
    logging.basicConfig(level=logging.WARNING)
    app = FastAPI()
    app.include_router(router)
    task = asyncio.create_task(analysis_worker())
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://benchmark") as client:
            response = await client.post("/jobs/submit", json={"pgn_string": request["case"]["pgn"],
                "commentary_level": request["level"], "llm_provider": request["provider"]})
            response.raise_for_status()
            job_id = response.json()["job_id"]
            while True:
                status_response = await client.get(f"/jobs/{job_id}/status")
                status_response.raise_for_status()
                status = status_response.json()
                if status["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)
            if status["status"] != "completed":
                raise RuntimeError("Annotation job failed; inspect worker.log")
            response = await client.get(f"/jobs/{job_id}/game")
            response.raise_for_status()
            write_json(folder / "game.json", response.json())
            export = await client.get(f"/jobs/{job_id}/pgn")
            export.raise_for_status()
            (folder / "game.pgn").write_text(export.text, encoding="utf-8")
            logs = folder / "logs/llm" / f"{job_id}.jsonl"
            calls = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()] if logs.exists() else []
            # Only an allowlist of configuration is recorded, never credentials.
            names = ("ENVISIONED_BASE_PLIES", "ENVISIONED_MAX_PLIES", "ENVISIONED_PROBE_TRIM_ENABLED",
                     "REFUTATION_SCAN", "REFUTATION_SCAN_MAX_POSITIONS", "LEAF_PROBE_DEPTH",
                     "REVERSE_ORDER_ENGINE_ANALYSIS", "CLAIM_DEDUP_WINDOW_PLIES", "STRUCTURAL_DEDUP_WINDOW_PLIES")
            write_json(folder / "runtime.json", {"seconds": time.perf_counter() - started,
                "job_id": job_id, "llm_calls": len(calls), "llm_failed_calls": sum(not c.get("ok") for c in calls),
                "tokens": sum(c.get("token_usage") or 0 for c in calls),
                "model": resolve_model(request["provider"], "composer", "key_moment"),
                "engine": engine.engine.id, "engine_sha256": sha(Path(engine.engine_path).read_bytes()),
                "threads": 1, "hash_mb": engine.hash_size, "use_nnue": engine.use_nnue,
                "environment_overrides": {key: os.environ[key] for key in names if key in os.environ}})
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        engine.close()


def export_ratings(out: Path, reports: list[dict], seed: int):
    rows, keys = [], []
    for report in reports:
        game_path = Path(report["artifact_dir"]) / "game.json"
        if report.get("error") or not game_path.exists():
            continue
        game = json.loads(game_path.read_text(encoding="utf-8"))
        for ply, move in enumerate(game["moves"], 1):
            if not move.get("final_comment"):
                continue
            sid = sha(f"{seed}:{report['case']}:{report['mode']}:{report['repeat']}:{ply}".encode())[:16]
            fen = game["metadata"].get("initial_fen") if ply == 1 else game["moves"][ply - 2]["fen"]
            rows.append({"sample_id": sid, "rater_id": "", "fen_before": fen, "move": move["san"],
                "comment": move["comment"], "correctness": "", "relevance": "", "usefulness": "", "concision": "", "notes": ""})
            keys.append({"sample_id": sid, "case": report["case"], "mode": report["mode"], "repeat": report["repeat"], "ply": ply})
    random.Random(seed).shuffle(rows)
    for filename, data, fields in (("ratings.csv", rows, ["sample_id", "rater_id", "fen_before", "move", "comment", "correctness", "relevance", "usefulness", "concision", "notes"]),
                                   ("rating_key.csv", keys, ["sample_id", "case", "mode", "repeat", "ply"])):
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)


def score_ratings(path: Path, key_path: Path):
    keys = {r["sample_id"]: r for r in csv.DictReader(key_path.open(encoding="utf-8"))}
    grouped, seen = {}, set()
    for row in csv.DictReader(path.open(encoding="utf-8")):
        pair = (row["sample_id"], row["rater_id"].strip())
        if pair[0] not in keys or not pair[1] or pair in seen:
            raise ValueError("Unknown sample, missing rater ID or duplicate sample/rater")
        seen.add(pair)
        group = grouped.setdefault(keys[pair[0]]["mode"], {})
        for metric in ("correctness", "relevance", "usefulness", "concision"):
            value = int(row[metric])
            if value not in range(1, 6):
                raise ValueError("Every rating must be an integer from 1 to 5")
            group.setdefault(metric, []).append(value)
    return {mode: {metric: {"n": len(v), "mean": statistics.mean(v)} for metric, v in group.items()} for mode, group in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks/chapter5-v1.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=["template", "llm", "both"], default="template")
    parser.add_argument("--provider", choices=["openai", "cursor", "grok"], default="openai")
    parser.add_argument("--level", choices=["beginner", "intermediate", "expert"], default="intermediate")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--case", action="append")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--audit-existing", type=Path, help="Review-directory CASE.json/CASE.pgn files; no engine or LLM calls")
    parser.add_argument("--baseline", type=Path, help="Earlier report.json with the same suite hash")
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--rating-key", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        asyncio.run(worker(args.worker.resolve()))
        return 0
    if args.ratings:
        if not args.rating_key:
            parser.error("--ratings requires --rating-key")
        print(json.dumps(score_ratings(args.ratings, args.rating_key), indent=2))
        return 0
    if not args.out or args.repeat < 1 or args.timeout <= 0:
        parser.error("--out, positive --repeat and positive --timeout are required")
    if args.audit_existing and (args.mode != "template" or args.repeat != 1):
        parser.error("--audit-existing audits each saved case once; do not combine it with --mode or --repeat")
    from app.evaluation import aggregate, audit_game
    suite = load_suite(args.suite.resolve(), args.case)
    previous = None
    if args.baseline:
        previous = json.loads(args.baseline.read_text(encoding="utf-8"))
        if previous.get("suite_sha256") != suite["sha256"]:
            raise ValueError("Baseline suite differs; compare the same cases and expectations")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier experiment
    write_json(out / "suite.json", suite)
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "app").rglob("*.py")):
        source_hash.update(path.relative_to(ROOT).as_posix().encode())
        source_hash.update(path.read_bytes())
    environment = {"python": sys.version, "platform": platform.platform(), "source_sha256": source_hash.hexdigest(),
                   "packages": {name: importlib.metadata.version(name) for name in ("chess", "pydantic", "httpx", "fastapi")}}
    reports = []
    modes = ["template", "llm"] if args.mode == "both" else [args.mode]
    for case in suite["cases"]:
        for mode in modes:
            for repeat in range(args.repeat):
                folder = out / "raw" / f"{case['id']}-{mode}-{repeat}"
                folder.mkdir(parents=True)
                record = {"case": case["id"], "category": case.get("category"), "mode": mode,
                          "repeat": repeat, "artifact_dir": str(folder), "error": None}
                try:
                    if args.audit_existing:
                        old = args.audit_existing.resolve()
                        (folder / "game.json").write_bytes((old / f"{case['id']}.json").read_bytes())
                        (folder / "game.pgn").write_bytes((old / f"{case['id']}.pgn").read_bytes())
                        record["mode"] = "existing"
                    else:
                        request = folder / "request.json"
                        write_json(request, {"case": case, "mode": mode, "level": args.level, "provider": args.provider})
                        # Capture worker output and preserve it on success or timeout.
                        try:
                            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(request)],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", timeout=args.timeout)
                        except subprocess.TimeoutExpired as exc:
                            captured = exc.stdout or b""
                            (folder / "worker.log").write_text(captured.decode("utf-8", errors="replace") if isinstance(captured, bytes) else captured, encoding="utf-8")
                            raise
                        (folder / "worker.log").write_text(proc.stdout or "", encoding="utf-8")
                        if proc.returncode:
                            raise RuntimeError(f"Worker exited {proc.returncode}; see worker.log")
                        record["runtime"] = json.loads((folder / "runtime.json").read_text(encoding="utf-8"))
                    data = json.loads((folder / "game.json").read_text(encoding="utf-8"))
                    record["audit"] = audit_game(case["pgn"], data, (folder / "game.pgn").read_text(encoding="utf-8"), case["required_comment_plies"])
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                reports.append(record)
                print(json.dumps({"case": case["id"], "mode": mode, "repeat": repeat,
                    "error": record["error"], "issues": len(record.get("audit", {}).get("issues", []))}), flush=True)
                write_json(out / "progress.json", reports)
    report = {"protocol": "annotation-benchmark-v1", "suite_sha256": suite["sha256"], "suite_version": suite["version"],
              "level": args.level, "provider": args.provider, "seed": args.seed, "environment": environment,
              "runs": reports, "aggregate": aggregate(reports),
              "by_mode": {mode: aggregate([r for r in reports if r["mode"] == mode]) for mode in sorted({r['mode'] for r in reports})}}
    llm_runs = [r for r in reports if r["mode"] == "llm"]
    report["llm_verification"] = {
        "requested_runs": len(llm_runs),
        "logical_calls": sum(r.get("runtime", {}).get("llm_calls", 0) for r in llm_runs),
        "failed_logical_calls": sum(r.get("runtime", {}).get("llm_failed_calls", 0) for r in llm_runs),
        "accepted_comments": sum(r.get("audit", {}).get("renderings", {}).get("llm", 0) for r in llm_runs),
        "note": "Calls count logical requests; provider retries are recorded in worker.log. Template fallbacks do not verify LLM operation.",
    }
    report["llm_verification"]["verified"] = bool(llm_runs) and report["llm_verification"]["accepted_comments"] > 0 and report["llm_verification"]["failed_logical_calls"] == 0
    if previous is not None:
        report["baseline"] = {"path": str(args.baseline.resolve()), "aggregate": previous["aggregate"]}
    write_json(out / "report.json", report)
    export_ratings(out, reports, args.seed)
    print(json.dumps(report["aggregate"], indent=2))
    return int(report["aggregate"]["failed_runs"] > 0 or any(c["passed"] != c["total"] for c in report["aggregate"]["checks"].values())
               or (bool(llm_runs) and not report["llm_verification"]["verified"]))


if __name__ == "__main__":
    raise SystemExit(main())
