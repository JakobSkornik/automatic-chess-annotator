"""Replay saved evidence for a small commentary regression/compatibility check.

No engine search is performed. The played continuation is limited to the saved
display line, so this is not a fresh whole-game benchmark or a quality estimate.
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chess

from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.llm_policy import resolve_model
from app.core.commentary.phases.composer import compose_facts_comment
from app.core.commentary.phases.composer.framing import comment_archetype
from app.core.commentary.phases.composer.template import render_facts_template
from app.core.commentary.rules.facts_builder import build_comment_facts
from app.models.chess_events import AnalyzedMoveData, MoveEvent


def ucis(fen, sans):
    board = chess.Board(fen)
    result = []
    for san in sans:
        move = board.parse_san(san)
        result.append(move.uci())
        board.push(move)
    return result


def rebuild(data, ply):
    move = data["moves"][ply - 1]
    saved = move["comment_facts"]
    debug = move["debug"]
    line = saved["display_line"]
    before = line["start_fen"]
    pv = ucis(before, line["san"])
    best_san = debug.get("best_move_san")
    best_uci = chess.Board(before).parse_san(best_san).uci() if best_san else None
    pvs = []
    for variation in move.get("variations") or []:
        score = variation.get("score") or {}
        cp = score.get("cp")
        if score.get("mate") is not None:
            mate = score["mate"]
            cp = (1 if mate >= 0 else -1) * (1000000 - abs(mate))
        pvs.append([SimpleNamespace(move=u, score=cp) for u in ucis(before, variation["line"])])
    event = MoveEvent(
        move_index=ply - 1, ply=ply, san=move["san"], uci=move["uci"],
        fen_before=before, fen_after=move["fen"], phase=move["phase"],
        eval_before_cp=debug.get("eval_before_cp"), eval_after_cp=debug.get("eval_after_cp"),
        move_quality=debug.get("move_quality") or "good", event_type="quiet",
        best_move_san=best_san, best_move_uci=best_uci,
        best_move_eval_cp=debug.get("best_move_eval_cp"),
        key_moment_type=debug.get("key_moment_type"),
    )
    row = AnalyzedMoveData(
        index=ply - 1, ply=ply, san=move["san"], uci=move["uci"],
        fen_before=before, fen_after=move["fen"], phase_raw=move["phase"], pvs=pvs,
        hidden_features={"_engine": {"after_pv_uci": pv[1:]}},
    )
    return build_comment_facts(row, event, depth=saved.get("depth") or 16)


async def run(args):
    if args.live:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        args.model = args.model or resolve_model(args.provider, "composer")
    raw = args.snapshot.read_bytes()
    data = json.loads(raw)
    result = {
        "scope": "development_smoke_replayed_saved_lines_not_fresh_engine_or_quality_benchmark",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "provider": args.provider if args.live else None, "model": args.model,
        "mode": args.mode, "effort": args.effort, "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def one(ply, service=None):
        facts = rebuild(data, ply)
        original = data["moves"][ply - 1]
        archetype = comment_archetype((original.get("debug") or {}).get("key_moment_type"))
        template = render_facts_template(facts, archetype=archetype)
        row = {"ply": ply, "san": facts.san, "before": original["comment"], "template": template,
               "facts": facts.model_dump(mode="json")}
        if service:
            started = time.perf_counter()
            rendered = await compose_facts_comment(
                service, facts, model=args.model, effort=args.effort,
                key_moment_type=(original.get("debug") or {}).get("key_moment_type"))
            row.update(result=rendered, elapsed_seconds=time.perf_counter() - started,
                       token_usage=service._last_token_usage or None,
                       request_metadata=service._provider.last_request)
            await service.record_annotation(dict(rendered, ply=ply, san=facts.san, uci=facts.uci))
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"ply": ply, "san": facts.san,
                          "rendering": row.get("result", {}).get("rendering", "template"),
                          "contract_ok": row.get("result", {}).get("contract_ok")}), flush=True)

    if args.live:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        os.environ["HUMANIZATION_LEVEL"] = "2"
        os.environ["COMMENTARY_RENDER_MODE"] = args.mode
        service = AdvancedCommentService(provider_key=args.provider)
        async with service.commentary_session("revision-smoke") as session:
            for ply in sorted(args.plies, reverse=True):
                await one(ply, session)
    else:
        for ply in sorted(args.plies, reverse=True):
            await one(ply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plies", type=int, nargs="+", default=[60, 54, 10])
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--provider", choices=["openai", "cursor", "grok"], default="cursor")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", choices=["low", "medium", "high"], default="low")
    parser.add_argument("--mode", choices=["grounded_rewrite", "bounded"], default="grounded_rewrite")
    asyncio.run(run(parser.parse_args()))
