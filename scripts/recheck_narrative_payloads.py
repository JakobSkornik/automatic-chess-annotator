"""Run the current narrative contract on frozen evidence from an LLM call log.

No engine rerun; selected cases are a development check, not a quality benchmark.
The resulting annotations are kept separate from the original whole-game run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.phases.composer import narrative


async def run(args):
    load_dotenv(ROOT / ".env")
    os.environ["HUMANIZATION_LEVEL"] = "2"
    records = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines()]
    inputs = {r["ply"]: r for r in records if r["schema_name"] == "grounded_rewrite"}
    result = {"scope": "Targeted live regeneration on frozen evidence; no engine rerun or whole-game quality estimate.",
              "writer_system_sha256": hashlib.sha256(narrative.SYSTEM.encode()).hexdigest(),
              "review_system_sha256": hashlib.sha256(narrative.REVIEW_SYSTEM.encode()).hexdigest(),
              "rows": []}
    service = AdvancedCommentService(provider_key="cursor")
    model = resolve_model("cursor", "composer", "key_moment")
    result["model"] = model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with service.commentary_session("narrative-final-focused-check") as session:
        for ply in sorted(args.plies, reverse=True):
            source = inputs[ply]
            payload = json.loads(source["user"])
            evidence = payload["evidence"]
            for branch in evidence["board_context"].values():
                branch["start"] = narrative._inventory(branch["start"]["fen"])
                for position in branch["positions"]:
                    position["board"] = narrative._inventory(position["board"]["fen"])
            user = json.dumps(payload, ensure_ascii=False)
            row = {"ply": ply, "source_payload_sha256": hashlib.sha256(source["user"].encode()).hexdigest(),
                   "submitted_payload_sha256": hashlib.sha256(user.encode()).hexdigest(),
                   "board_context_policy": "current mechanical inventory only; saved claims and traces unchanged",
                   "original_draft": source["response"]}
            started = time.perf_counter()
            try:
                raw = await session._llm_call_json_schema(narrative.SYSTEM, user, model=model,
                    effort="low", schema=narrative.SCHEMA, schema_name="grounded_rewrite", max_output_tokens=1200)
                row["raw_proposal"] = raw
                proposal, reviews = narrative.parse_proposal(raw, evidence)
                review = await session.review_json(narrative.REVIEW_SYSTEM, json.dumps({"sentences": reviews}),
                    model=model, effort="low", schema=narrative.REVIEW_SCHEMA,
                    schema_name="grounded_rewrite_review", max_output_tokens=1500)
                row["raw_review"] = review
                row["review_decisions"] = narrative.validate_review(review, len(reviews))
                row.update(text=narrative.render(proposal, evidence), accepted=True, selected_claims=proposal)
                await session.record_annotation({"text": row["text"], "ply": ply, "contract_ok": True,
                                                 "rendering": "llm", "rendering_contract": narrative.VERSION})
            except Exception as exc:
                row.update(accepted=False, reason=str(exc))
                await session.record_annotation({"text": "", "ply": ply, "contract_ok": False, "rendering": "template"})
            row["seconds"] = time.perf_counter() - started
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"ply": ply, "accepted": row["accepted"], "seconds": row["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plies", type=int, nargs="+", default=[71, 63, 55])
    asyncio.run(run(parser.parse_args()))
