"""Frozen-evidence component experiment; never uses future recorded moves."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.pgn

from app.evaluation import audit_game, parse_game
from app.models.chess_events import AnalyzedMoveData, MoveEvent
from app.models.comment_facts import CommentFacts

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = [
    {"id": "B00", "horizon": "current_quiet_bounded", "claims": "current", "renderer": "template"},
    {"id": "B10", "horizon": "adaptive_feature", "claims": "current", "renderer": "template"},
    {"id": "B01", "horizon": "current_quiet_bounded", "claims": "contrastive", "renderer": "template"},
    {"id": "B11", "horizon": "adaptive_feature", "claims": "contrastive", "renderer": "template"},
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def source_files():
    paths = list((ROOT / "app").rglob("*.py")) + list((ROOT / "scripts").glob("*.py"))
    paths += list(ROOT.glob("requirements*.txt"))
    paths += list((ROOT / "app/research").glob("*.html"))
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in sorted(paths)}


def verify_manifest(folder):
    manifest = read(folder / "manifest.json")
    if digest(manifest) != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Experiment manifest changed after preparation")
    if digest(read(folder / "source.json")) != manifest["source_hash"]:
        raise ValueError("Archived source snapshot changed")
    return manifest


def prepare(suite_path, folder, *, depth=16, probe_depth=8, max_plies=12, extra_targets=True, study_path=None):
    """Create a diagnostic component corpus, not a held-out research dataset."""
    from scripts.benchmark_annotations import load_suite
    folder = Path(folder).resolve()
    if depth < 1 or probe_depth < 1 or max_plies < 1:
        raise ValueError("Search depths and horizon must be positive")
    study = None
    if study_path is not None:
        from app.research.study import load_study
        study = load_study(Path(study_path))
    suite = load_suite(Path(suite_path), None) if study is None else {"cases": []}
    cases = []
    extras = {"opera": [13, 19], "pawn_endgame": [1, 5], "quiet_opening": [20]} if extra_targets else {}
    for source in suite["cases"]:
        game = parse_game(source["pgn"])
        moves = list(game.mainline_moves())
        targets = sorted(set(source["required_comment_plies"] + extras.get(source["id"], [])))
        for ply in targets:
            if not 1 <= ply <= len(moves):
                raise ValueError(f"Invalid target {source['id']}:{ply}")
            # Only history through the target is retained. No original future or annotations.
            clean = chess.pgn.Game()
            clean.setup(game.board())
            clean.headers["Result"] = "*"
            node = clean
            for move in moves[:ply]:
                node = node.add_variation(move)
            text = clean.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))
            board = game.board()
            for move in moves[:ply - 1]:
                board.push(move)
            cid = f"{source['id']}-p{ply}"
            cases.append({"id": cid, "category": source.get("category", "diagnostic"),
                "source_group": source["id"], "family_group": source["id"], "split": "diagnostic",
                "initial_fen": game.board().fen(), "target_ply": ply, "fen_before": board.fen(),
                "target_uci": moves[ply - 1].uci(), "pgn": text, "input_hash": digest(text),
                "teaching_target": ply in source["required_comment_plies"],
                "provenance": {"suite_version": suite["version"], "source_case": source["id"]}})
    if study is not None:
        cases = study["cases"]
    if not cases:
        raise ValueError("No target positions selected")
    dataset_hash = digest(cases)
    sources = source_files()
    manifest = {"protocol": "thesis-component-v1", "experiment_id": folder.name,
        "dataset_hash": dataset_hash, "source_hash": digest(sources), "track": "component",
        "dataset_status": "diagnostic_not_held_out", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audience": "intermediate", "seed": 20260905, "replicates": 1,
        "engine": {"depth": depth, "probe_depth": probe_depth, "max_plies": max_plies, "base_plies": 5, "multipv": 2, "threads": 1},
        "variants": VARIANTS, "cases": cases,
        "environment": {"python": sys.version, "platform": platform.platform()}}
    manifest["rubric_version"] = "v1"
    manifest["classification_adapter"] = "component-v1: best move identity; 50/100/200 cp loss; endgame <=10 pieces; forced target"
    manifest["scope"] = "Selector comparison on shared engine evidence; not full application selection/classification parity"
    if study is not None:
        manifest["rater_ids"] = study["evaluation"]["rater_ids"]
        manifest["evaluation"] = study["evaluation"]
        for key in ("engine", "audience", "seed", "replicates", "rubric_version", "dataset_status", "variants", "corpus_hash", "split_audit", "rater_ids"):
            if key in study:
                manifest[key] = study[key]
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "source.json", sources)
    write_new(folder / "manifest.json", manifest)
    (folder / "manifest.sha256").write_text(digest(manifest), encoding="utf-8")
    for case in cases:
        write_new(folder / "inputs" / f"{case['id']}.json", case)
    return {"experiment_id": folder.name, "cases": len(cases), "scheduled_runs": len(cases) * len(manifest["variants"]) * manifest["replicates"]}


def configure(manifest):
    os.environ["ENVISIONED_BASE_PLIES"] = str(manifest["engine"]["base_plies"])
    os.environ["ENVISIONED_MAX_PLIES"] = str(manifest["engine"]["max_plies"])
    os.environ["ENVISIONED_PROBE_TRIM_ENABLED"] = "1"
    os.environ["HUMANIZATION_LEVEL"] = "0"


def load_snapshot(folder, case_id):
    snap = read(folder / "snapshots" / f"{case_id}.json")
    expected = snap["snapshot_hash"]
    if digest({k: v for k, v in snap.items() if k != "snapshot_hash"}) != expected:
        raise ValueError("Snapshot hash mismatch")
    return snap


def replay_inputs(snap):
    row_data = dict(snap["row"])
    row_data["pvs"] = [[SimpleNamespace(**move) for move in line] for line in row_data["pvs"]]
    def probe(fen):
        if fen not in snap["probes"]:
            raise ValueError("Selector requested evidence outside the frozen snapshot")
        return snap["probes"][fen]
    return AnalyzedMoveData(**row_data), MoveEvent(**snap["event"]), probe


def snapshot(folder):
    from app.core.engine.engine_connector import get_global_engine_connector
    from app.core.commentary.rules.facts_builder import build_comment_facts
    from app.core.commentary.features.guid_features import compute_feature_vector_fen
    folder = Path(folder).resolve()
    manifest = verify_manifest(folder)
    if digest(source_files()) != manifest["source_hash"]:
        raise ValueError("Source changed; prepare a new experiment before computing evidence")
    configure(manifest)
    config = manifest["engine"]
    engine = get_global_engine_connector()
    engine.engine.configure({"Threads": config["threads"]})
    outcomes = []
    try:
        for case in manifest["cases"]:
            if (folder / "snapshots" / f"{case['id']}.json").exists():
                load_snapshot(folder, case["id"])
                outcomes.append({"case": case["id"], "status": "verified_existing"})
                continue
            started = time.perf_counter()
            try:
                engine.engine.configure({"Clear Hash": None})
                game = parse_game(case["pgn"])
                board = game.board()
                history = list(game.mainline_moves())
                for move in history[:-1]:
                    board.push(move)
                played = history[-1]
                after = board.copy()
                after.push(played)
                calls = []
                def analyse(b, depth, count=1):
                    info = engine.analyse(b, depth=depth, multiPv=count)
                    if isinstance(info, dict):
                        info = [info]
                    calls.append({"fen": b.fen(), "depth_requested": depth,
                        "results": [{"depth": x.get("depth"), "nodes": x.get("nodes"),
                            "time": x.get("time"), "score_cp": x["score"].white().score(mate_score=1000000),
                            "pv": [m.uci() for m in x.get("pv", [])]} for x in info]})
                    return calls[-1]["results"]
                pre = analyse(board, config["depth"], config["multipv"])
                post = analyse(after, config["depth"])
                best = pre[0]
                played_score = post[0]["score_cp"]
                best_move = best["pv"][0]
                sign = 1 if board.turn else -1
                loss = sign * (best["score_cp"] - played_score)
                quality = "best" if best_move == played.uci() else ("blunder" if loss >= 200 else "mistake" if loss >= 100 else "inaccuracy" if loss >= 50 else "good")
                phase = "end" if len(board.piece_map()) <= 10 else "mid"
                event = MoveEvent(move_index=case["target_ply"] - 1, ply=case["target_ply"],
                    san=board.san(played), uci=played.uci(), fen_before=board.fen(), fen_after=after.fen(),
                    phase=phase, eval_before_cp=best["score_cp"], eval_after_cp=played_score,
                    move_quality=quality, event_type="quiet", best_move_uci=best_move,
                    best_move_san=board.san(chess.Move.from_uci(best_move)), best_move_eval_cp=best["score_cp"])
                row = {"index": case["target_ply"] - 1, "ply": case["target_ply"], "san": event.san,
                    "uci": event.uci, "fen_before": board.fen(), "fen_after": after.fen(), "score_cp": played_score,
                    "phase_raw": phase, "pvs": [[{"move": u, "score": line["score_cp"]} for u in line["pv"]] for line in pre],
                    "hidden_features": {"_engine": {"after_pv_uci": post[0]["pv"]}}}
                probes, features = {}, {}
                for line in [[played.uci(), *post[0]["pv"]], *[x["pv"] for x in pre]]:
                    b = board.copy()
                    for uci in [None, *line[:config["max_plies"]]]:
                        if uci is not None:
                            b.push_uci(uci)
                        fen = b.fen()
                        if fen not in probes:
                            probes[fen] = analyse(b, config["probe_depth"])[0]["score_cp"]
                            features[fen] = {k: v.model_dump() for k, v in compute_feature_vector_fen(fen).items()}
                snap = {"case_id": case["id"], "dataset_hash": manifest["dataset_hash"], "row": row,
                    "event": event.model_dump(mode="json"), "probes": probes, "features": features,
                    "searches": calls, "engine": {"id": engine.engine.id,
                        "binary_sha256": hashlib.sha256(Path(engine.engine_path).read_bytes()).hexdigest(),
                        "hash_mb": engine.hash_size, "use_nnue": engine.use_nnue, **config}}
                replay_row, replay_event, probe = replay_inputs(snap)
                snap["baseline_facts"] = build_comment_facts(replay_row, replay_event, depth=config["depth"], prober=probe).model_dump(mode="json")
                snap["seconds"] = time.perf_counter() - started
                snap["snapshot_hash"] = digest(snap)
                write_new(folder / "snapshots" / f"{case['id']}.json", snap)
                result = {"case": case["id"], "status": "completed", "seconds": round(snap["seconds"], 3)}
                outcomes.append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                failure_dir = folder / "snapshot-attempts" / case["id"]
                attempt = len(list(failure_dir.glob("*.json")))
                failure = {"case_id": case["id"], "attempt": attempt, "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}", "seconds": time.perf_counter() - started}
                write_new(failure_dir / f"a{attempt}.json", failure)
                outcomes.append(failure)
                print(json.dumps(failure), flush=True)
    finally:
        engine.close()
    return {"snapshots": len(outcomes), "failed": sum(x["status"] == "failed" for x in outcomes)}


def audit_sample(sample, case):
    """Reuse the production-artifact audit on an exact one-move target game."""
    from app.core.engine.analysis.serialization import _facts_to_json
    from app.core.io.pgn_writer import game_json_to_pgn
    from app.models.GameJson import GameJson, GameMetadata, GameMove, AnalysisInfo
    facts = CommentFacts(**sample["facts"])
    before = chess.Board(sample["fen_before"])
    board = before.copy()
    board.push_uci(facts.uci)
    game = chess.pgn.Game()
    game.setup(before)
    game.add_variation(chess.Move.from_uci(facts.uci))
    game.headers["Result"] = board.result()
    source = str(game)
    rendered = GameJson(metadata=GameMetadata(id=sample["sample_id"], white="White", black="Black",
        result=board.result(), initial_fen=before.fen()),
        moves=[GameMove(mn=before.fullmove_number, color="w" if before.turn else "b", san=facts.san,
            uci=facts.uci, fen=board.fen(), comment=sample["comment"], final_comment=True,
            comment_facts=_facts_to_json(facts))], commentary_complete=True,
        analysis_info=AnalysisInfo(engine="Stockfish", depth=facts.depth or 16, multipv=2, timestamp=0))
    pgn = game_json_to_pgn(rendered)
    report = audit_game(source, rendered.model_dump(), pgn, [1])
    sample["pgn"] = pgn
    return report


def run(folder, *, resume=False, retry_failed=False):
    from app.core.commentary.rules.facts_builder import build_comment_facts
    from app.core.commentary.phases.composer import _audience_view, render_facts_template
    folder = Path(folder).resolve()
    manifest = verify_manifest(folder)
    if digest(source_files()) != manifest["source_hash"]:
        raise ValueError("Source changed since preparation; create a new experiment")
    configure(manifest)
    schedule = [{"run_id": f"{manifest['experiment_id']}--{c['id']}-{v['id']}-r{rep}-a0", "case_id": c["id"], "variant_id": v["id"]}
                for c in manifest["cases"] for v in manifest["variants"] for rep in range(manifest["replicates"])]
    random.Random(manifest["seed"]).shuffle(schedule)
    if not (folder / "schedule.json").exists():
        write_new(folder / "schedule.json", schedule)
    elif read(folder / "schedule.json") != schedule:
        raise ValueError("Scheduled matrix changed")
    outcomes = []
    for cell in schedule:
        cell = dict(cell)
        replicate = int(cell["run_id"].rsplit("-r", 1)[1].split("-a")[0])
        attempt = 0
        stem = cell["run_id"].rsplit("-a", 1)[0]
        while (folder / "runs" / f"{stem}-a{attempt + 1}" / "result.json").exists():
            attempt += 1
        cell["run_id"] = f"{stem}-a{attempt}"
        dest = folder / "runs" / cell["run_id"] / "result.json"
        if dest.exists():
            if not (resume or retry_failed):
                raise FileExistsError("Run already exists; use --resume to verify and skip")
            saved = read(dest)
            if digest(saved) != dest.with_suffix(".sha256").read_text().strip():
                raise ValueError("Saved run was modified")
            if saved["status"] == "completed" or not retry_failed:
                outcomes.append(saved)
                continue
            attempt += 1
            cell["run_id"] = f"{stem}-a{attempt}"
            dest = folder / "runs" / cell["run_id"] / "result.json"
        case = next(c for c in manifest["cases"] if c["id"] == cell["case_id"])
        variant = next(v for v in manifest["variants"] if v["id"] == cell["variant_id"])
        result = {**cell, "experiment_id": manifest["experiment_id"], "replicate": replicate, "attempt": attempt,
                  "status": "running", "requested_renderer": "template", "actual_renderer": "template", "samples": [], "error": None}
        started = time.perf_counter()
        try:
            snap = load_snapshot(folder, case["id"])
            result["snapshot_hash"] = snap["snapshot_hash"]
            row, event, probe = replay_inputs(snap)
            facts_started = time.perf_counter()
            facts = build_comment_facts(row, event, depth=manifest["engine"]["depth"], prober=probe,
                horizon_policy=variant["horizon"], claim_policy=variant["claims"])
            if facts is None:
                raise ValueError("Forced component target did not produce facts")
            if variant["id"] == "B00" and facts.model_dump(mode="json") != snap["baseline_facts"]:
                raise ValueError("B00 fails baseline fact parity on frozen evidence")
            facts_seconds = time.perf_counter() - facts_started
            render_started = time.perf_counter()
            text = render_facts_template(_audience_view(facts, manifest["audience"]))
            render_seconds = time.perf_counter() - render_started
            sample = {"sample_id": digest(cell)[:24], "target_ply": case["target_ply"],
                "fen_before": case["fen_before"], "move_san": event.san, "comment": text,
                "facts": facts.model_dump(mode="json")}
            audit_started = time.perf_counter()
            sample["audit"] = audit_sample(sample, case)
            result["stage_seconds"] = {"facts": facts_seconds, "render": render_seconds, "audit": time.perf_counter() - audit_started,
                                       "shared_snapshot": snap.get("seconds")}
            result.update(status="completed", explanation_outcome="delivered", samples=[sample])
        except Exception as exc:
            result.update(status="failed", explanation_outcome="failed", error_category=type(exc).__name__, error=f"{type(exc).__name__}: {exc}")
        result["seconds"] = time.perf_counter() - started
        write_new(dest, result)
        dest.with_suffix(".sha256").write_text(digest(result), encoding="utf-8")
        outcomes.append(result)
        print(json.dumps({"run": cell["run_id"], "status": result["status"], "error": result["error"]}), flush=True)
    return {"runs": len(outcomes), "failed": sum(r["status"] != "completed" for r in outcomes),
            "audit_issues": sum(len(s["audit"]["issues"]) for r in outcomes for s in r["samples"])}


def audit(folder):
    folder = Path(folder).resolve()
    manifest = verify_manifest(folder)
    cases = {c["id"]: c for c in manifest["cases"]}
    issues = []
    schedule = read(folder / "schedule.json")
    for cell in schedule:
        try:
            stem = cell["run_id"].rsplit("-a", 1)[0]
            attempts = list((folder / "runs").glob(f"{stem}-a*/result.json"))
            if attempts:
                cell = {**cell, "run_id": max(attempts, key=lambda p: int(p.parent.name.rsplit("-a", 1)[1])).parent.name}
            path = folder / "runs" / cell["run_id"] / "result.json"
            result = read(path)
            if digest(result) != path.with_suffix(".sha256").read_text().strip():
                raise ValueError("Run hash mismatch")
            if result["status"] != "completed":
                raise ValueError(result.get("error") or result["status"])
            snap = load_snapshot(folder, cell["case_id"])
            if result["snapshot_hash"] != snap["snapshot_hash"]:
                raise ValueError("Run evidence does not match frozen snapshot")
            for sample in result["samples"]:
                issues.extend({"run": cell["run_id"], **i} for i in audit_sample(sample, cases[cell["case_id"]])["issues"])
        except Exception as exc:
            issues.append({"run": cell["run_id"], "detail": str(exc)})
    return {"scheduled_runs": len(schedule), "issues": issues}


async def renderer_check(folder, *, provider="cursor", limit=2):
    """Small explicitly requested provider smoke test on frozen B11 facts."""
    from app.core.commentary.llm_providers import make_llm_provider
    from app.core.commentary.llm_policy import resolve_model
    from app.core.commentary.phases.composer import compose_facts_comment
    folder = Path(folder).resolve()
    manifest = verify_manifest(folder)
    service_provider = make_llm_provider(provider)
    if not service_provider.is_configured():
        raise ValueError(f"{provider} provider is not configured")
    destination = folder / "renderer-check"
    destination.mkdir(exist_ok=False)
    os.environ["HUMANIZATION_LEVEL"] = "2"
    logs = []
    class Service:
        _provider = service_provider
        async def _llm_call_json_schema(self, system, user, **kwargs):
            raw, usage = await asyncio.wait_for(service_provider.json_schema_call(system, user, **kwargs), timeout=120)
            logs.append({"system": system, "user": user, "response": raw, "tokens": usage})
            return raw
    output = []
    for path in sorted((folder / "runs").glob("*-B11-r0-a0/result.json")):
        result = read(path)
        for sample in result["samples"]:
            facts = CommentFacts(**sample["facts"])
            if facts.terminal_result or not facts.claims:
                continue
            rendered = await compose_facts_comment(Service(), facts, model=resolve_model(provider, "composer"),
                effort="low", level=manifest["audience"])
            output.append({"sample_id": sample["sample_id"], "case_id": result["case_id"], **rendered})
            if len(output) >= limit:
                break
        if len(output) >= limit:
            break
    write_new(destination / "report.json", {"provider": provider, "model": resolve_model(provider, "composer"),
        "outputs": output, "calls": len(logs), "tokens": sum(x["tokens"] or 0 for x in logs)})
    write_new(destination / "calls.json", logs)
    return {"calls": len(logs), "outputs": len(output), "accepted": sum(o["rendering"] == "llm" for o in output)}
