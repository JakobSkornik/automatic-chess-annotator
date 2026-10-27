"""Run a paired claims-only vs board-grounded bounded-editor prompt ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.research import prompt_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--provider", choices=("openai", "cursor", "grok"))
    parser.add_argument("--model")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=1500)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Run only the first N shuffled cells (smoke runs).")
    args = parser.parse_args()
    folder = args.experiment.resolve()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.stage == "prepare":
        if args.parent is None or args.provider is None or not args.model:
            parser.error("prepare requires --parent, --provider, and --model")
        result = prompt_ablation.prepare(args.parent, folder, provider=args.provider, model=args.model,
            effort=args.effort, max_output_tokens=args.max_output_tokens,
            replicates=args.replicates, seed=args.seed)
    elif args.stage == "run":
        result = prompt_ablation.run(folder, resume=args.resume, limit=args.limit)
    else:
        result = prompt_ablation.report(folder)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
