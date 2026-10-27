"""CLI for frozen component experiments. See docs/research-pipeline.md."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["all", "prepare", "snapshot", "run", "audit", "ingest", "packet", "ratings-import", "analyze", "secondary", "evaluator", "evaluator-serve", "renderer-check"])
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmarks/chapter5-v1.json")
    parser.add_argument("--study", type=Path, help="Validated executable study configuration")
    parser.add_argument("--rubric-version", help="Defaults to the frozen manifest rubric")
    parser.add_argument("--retry-failed", action="store_true", help="Append a new attempt for failed cells")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database", type=Path, default=ROOT / "data/research/results.sqlite")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ratings", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--probe-depth", type=int, default=8)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from app.research import runner
    folder = args.experiment.resolve()
    if args.rubric_version is None:
        args.rubric_version = runner.verify_manifest(folder).get("rubric_version", "v1") if folder.exists() else (runner.read(args.study).get("rubric_version", "v1") if args.study else "v1")
    if args.command == "all":
        from app.research.bank import ingest, packet
        from app.research.analysis import analyze
        if folder.exists() and not args.resume:
            parser.error("Existing experiments require --resume")
        if not folder.exists():
            runner.prepare(args.suite.resolve(), folder, depth=args.depth, probe_depth=args.probe_depth, study_path=args.study)
        runner.snapshot(folder)
        execution = runner.run(folder, resume=args.resume, retry_failed=args.retry_failed)
        audited = runner.audit(folder)
        indexed = ingest(folder, args.database.resolve())
        ratings = packet(folder, folder / "ratings") if not (folder / "ratings").exists() else {"existing": str(folder / "ratings")}
        analysis = analyze(folder, args.database.resolve(), args.out, rubric_version=args.rubric_version)
        from app.research.secondary import analyze_secondary
        from app.research.evaluator import build_evaluator
        secondary = analyze_secondary(folder, args.database.resolve(), (args.out or folder / "analysis") / "secondary", rubric_version=args.rubric_version)
        evaluator_path = folder / "evaluator"
        manifest = runner.verify_manifest(folder)
        evaluator = build_evaluator(folder, evaluator_path, rater_ids=tuple(manifest.get("rater_ids", ["evaluator-a", "evaluator-b"])), seed=manifest["seed"]) if not evaluator_path.exists() else {"existing": str(evaluator_path)}
        result = {"experiment_id": folder.name, **execution, "issues": audited["issues"],
                  "bank": indexed, "ratings": ratings, "analysis": analysis, "secondary": secondary, "evaluator": evaluator}
    elif args.command == "prepare":
        result = runner.prepare(args.suite.resolve(), folder, depth=args.depth, probe_depth=args.probe_depth, study_path=args.study)
    elif args.command == "snapshot":
        result = runner.snapshot(folder)
    elif args.command == "run":
        result = runner.run(folder, resume=args.resume, retry_failed=args.retry_failed)
    elif args.command == "audit":
        result = runner.audit(folder)
    elif args.command == "ingest":
        from app.research.bank import ingest
        result = ingest(folder, args.database.resolve())
    elif args.command == "packet":
        from app.research.bank import packet
        result = packet(folder, args.out or folder / "ratings")
    elif args.command == "ratings-import":
        if not args.ratings or not args.key:
            parser.error("ratings-import requires --ratings and --key")
        from app.research.bank import import_ratings
        result = import_ratings(args.database.resolve(), args.ratings, args.key, rubric_version=args.rubric_version)
    elif args.command == "analyze":
        from app.research.analysis import analyze
        result = analyze(folder, args.database.resolve(), args.out, rubric_version=args.rubric_version)
    elif args.command == "secondary":
        from app.research.secondary import analyze_secondary
        result = analyze_secondary(folder, args.database.resolve(), args.out or folder / "secondary", rubric_version=args.rubric_version)
    elif args.command == "evaluator":
        from app.research.evaluator import build_evaluator
        manifest = runner.verify_manifest(folder)
        result = build_evaluator(folder, args.out or folder / "evaluator", rater_ids=tuple(manifest.get("rater_ids", ["evaluator-a", "evaluator-b"])), seed=manifest["seed"])
    elif args.command == "evaluator-serve":
        from app.research.evaluator import serve_evaluator
        serve_evaluator(args.out or folder / "evaluator", port=args.port)
        return 0
    else:
        result = asyncio.run(runner.renderer_check(folder, limit=args.limit))
    print(json.dumps(result, indent=2, default=str))
    return int(bool(result.get("failed") or result.get("audit_issues") or result.get("issues")))


if __name__ == "__main__":
    raise SystemExit(main())
