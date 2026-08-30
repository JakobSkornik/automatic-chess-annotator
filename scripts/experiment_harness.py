"""Experiment harness: run the real annotation pipeline (engine + LLM) over a
batch of human-annotated games from data/pgn/perle.pgn, and measure how much
the generated commentary differs from the original human annotations.

Usage (from the automatic-chess-annotator repo root, with the venv active and
OPENAI_API_KEY / ANTHROPIC_API_KEY / stockfish available):

    python scripts/experiment_harness.py --n-games 20
    python scripts/experiment_harness.py --n-games 1 --start 0   # smoke test

Writes, under data/experiments/perle_harness/:
    raw/game_XX_full.json    - full GameJson produced by the pipeline
    raw/game_XX_human.json   - extracted human per-ply comments + NAGs
    raw/game_XX_diff.json    - per-ply comparison metrics
    progress.jsonl           - one line appended per finished/failed game
    summary.json             - aggregate stats once the batch finishes (or is
                               (re)computed on demand from whatever raw/*
                               files already exist via --summarize-only)
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import io
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field

import chess.pgn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.commentary.advanced_comment_service import AdvancedCommentService
from app.core.commentary.pipeline.game_pipeline import GameAnnotationPipeline
from app.core.engine.analysis_retriever import (
    assemble_game_json,
    run_engine_analysis_to_json,
)
from app.core.engine.engine_connector import get_global_engine_connector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("experiment_harness")

PGN_PATH = os.path.join("data", "pgn", "perle.pgn")
OUT_DIR = os.path.join("data", "experiments", "perle_harness")
RAW_DIR = os.path.join(OUT_DIR, "raw")

NAG_SYMBOL = {1: "!", 2: "?", 3: "!!", 4: "??", 5: "!?", 6: "?!"}
POSITIVE_SYMBOLS = {"!", "!!", "!?"}
NEGATIVE_SYMBOLS = {"?", "??", "?!"}

_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOPWORDS = {
    "the", "a", "an", "is", "was", "of", "to", "and", "in", "on", "with", "for",
    "this", "that", "it", "his", "her", "its", "as", "at", "by", "be", "are",
    "but", "not", "if", "or", "than", "then", "so", "now", "after", "before",
    "white", "black",
}


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _similarity(a: str, b: str) -> dict:
    if not a or not b:
        return {"ratio": 0.0, "jaccard": 0.0}
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        jac = 0.0
    else:
        jac = len(ta & tb) / len(ta | tb)
    return {"ratio": round(ratio, 4), "jaccard": round(jac, 4)}


@dataclass
class HumanPly:
    ply: int
    san: str
    comment: str
    nags: list[int] = field(default_factory=list)

    @property
    def symbol(self) -> str | None:
        for n in self.nags:
            if n in NAG_SYMBOL:
                return NAG_SYMBOL[n]
        return None


def load_games(path: str, max_read: int) -> list[chess.pgn.Game]:
    games = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while len(games) < max_read:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games


def extract_human_plies(game: chess.pgn.Game) -> list[HumanPly]:
    out = []
    node = game
    ply = 0
    while node.variations:
        nxt = node.variations[0]
        ply += 1
        comment = (nxt.comment or "").strip()
        out.append(
            HumanPly(
                ply=ply,
                san=node.board().san(nxt.move),
                comment=comment,
                nags=sorted(nxt.nags),
            )
        )
        node = nxt
    return out


def game_to_pgn_string(game: chess.pgn.Game) -> str:
    """Re-export headers + mainline moves only, stripped of human comments/NAGs
    so nothing about the original annotation can leak into the pipeline."""
    exporter = chess.pgn.StringExporter(
        headers=True, variations=False, comments=False
    )
    stripped = chess.pgn.Game()
    stripped.headers = game.headers.copy()
    node = stripped
    src = game
    while src.variations:
        src = src.variations[0]
        node = node.add_variation(src.move)
    return stripped.accept(exporter)


async def analyze_one_game(idx: int, game: chess.pgn.Game, llm_provider: str | None) -> dict:
    pgn_str = game_to_pgn_string(game)
    human_plies = extract_human_plies(game)

    engine = get_global_engine_connector()

    async def _progress(_pct: float, _msg: str) -> None:
        return None

    _, state = await run_engine_analysis_to_json(
        pgn_str, engine, _progress, metadata_id=f"perle_{idx:02d}"
    )

    advanced = AdvancedCommentService(provider_key=llm_provider)

    async def _commentary_cb(_t: str, _p: dict) -> None:
        return None

    await GameAnnotationPipeline().run_llm_phases(
        state,
        advanced,
        progress_callback=_progress,
        commentary_callback=_commentary_cb,
    )
    game_json = assemble_game_json(state)

    return {
        "idx": idx,
        "headers": dict(game.headers),
        "n_plies_human": len(human_plies),
        "n_plies_system": len(game_json.moves),
        "human_plies": [h.__dict__ for h in human_plies],
        "game_json": json.loads(game_json.model_dump_json()),
    }


def compare_game(record: dict) -> dict:
    human_by_ply = {h["ply"]: h for h in record["human_plies"]}
    moves = record["game_json"]["moves"]
    per_ply = []
    for i, mv in enumerate(moves):
        ply = i + 1
        h = human_by_ply.get(ply)
        sys_comment = (mv.get("comment") or "").strip()
        sys_symbol = mv.get("annotation")
        entry = {
            "ply": ply,
            "san": mv.get("san"),
            "human_san": h["san"] if h else None,
            "san_match": (h is not None and h["san"] == mv.get("san")),
            "human_comment": h["comment"] if h else "",
            "human_symbol": (
                NAG_SYMBOL.get(next((n for n in h["nags"] if n in NAG_SYMBOL), -1))
                if h else None
            ),
            "system_comment": sys_comment,
            "system_symbol": sys_symbol,
            "is_key_moment": mv.get("is_key_moment", False),
            "final_comment": mv.get("final_comment", False),
        }
        has_human = bool(h and h["comment"])
        has_system = bool(sys_comment)
        entry["both_present"] = has_human and has_system
        entry["human_only"] = has_human and not has_system
        entry["system_only"] = has_system and not has_human
        entry["neither"] = not has_human and not has_system
        if has_human and has_system:
            entry.update({"text_" + k: v for k, v in _similarity(h["comment"], sys_comment).items()})
        if entry["human_symbol"] and entry["system_symbol"]:
            hs, ss = entry["human_symbol"], entry["system_symbol"]
            if hs == ss:
                entry["symbol_agreement"] = "exact"
            elif (hs in POSITIVE_SYMBOLS) == (ss in POSITIVE_SYMBOLS):
                entry["symbol_agreement"] = "same_polarity"
            else:
                entry["symbol_agreement"] = "disagree"
        per_ply.append(entry)

    n = len(per_ply)
    both = [p for p in per_ply if p["both_present"]]
    game_summary = {
        "idx": record["idx"],
        "white": record["headers"].get("White"),
        "black": record["headers"].get("Black"),
        "n_plies": n,
        "n_human_annotated": sum(1 for p in per_ply if p["human_comment"]),
        "n_system_commented": sum(1 for p in per_ply if p["system_comment"]),
        "n_key_moments": sum(1 for p in per_ply if p["is_key_moment"]),
        "n_both_present": len(both),
        "n_human_only": sum(1 for p in per_ply if p["human_only"]),
        "n_system_only": sum(1 for p in per_ply if p["system_only"]),
        "avg_text_ratio": round(sum(p["text_ratio"] for p in both) / len(both), 4) if both else None,
        "avg_text_jaccard": round(sum(p["text_jaccard"] for p in both) / len(both), 4) if both else None,
        "symbol_pairs": sum(1 for p in per_ply if p.get("symbol_agreement")),
        "symbol_exact": sum(1 for p in per_ply if p.get("symbol_agreement") == "exact"),
        "symbol_same_polarity": sum(1 for p in per_ply if p.get("symbol_agreement") == "same_polarity"),
        "symbol_disagree": sum(1 for p in per_ply if p.get("symbol_agreement") == "disagree"),
    }
    return {"game_summary": game_summary, "per_ply": per_ply}


def _progress_line(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


async def run_batch(n_games: int, start: int, llm_provider: str | None) -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    progress_path = os.path.join(OUT_DIR, "progress.jsonl")

    all_games = load_games(PGN_PATH, start + n_games * 3 + 5)
    candidates = all_games[start:]
    done = 0
    attempted = 0
    for game in candidates:
        if done >= n_games:
            break
        attempted += 1
        idx = start + attempted - 1
        t0 = time.time()
        try:
            record = await analyze_one_game(idx, game, llm_provider)
            diff = compare_game(record)
            with open(os.path.join(RAW_DIR, f"game_{idx:02d}_full.json"), "w", encoding="utf-8") as f:
                json.dump(record["game_json"], f, indent=2)
            with open(os.path.join(RAW_DIR, f"game_{idx:02d}_human.json"), "w", encoding="utf-8") as f:
                json.dump(record["human_plies"], f, indent=2)
            with open(os.path.join(RAW_DIR, f"game_{idx:02d}_diff.json"), "w", encoding="utf-8") as f:
                json.dump(diff, f, indent=2)
            elapsed = round(time.time() - t0, 1)
            _progress_line(progress_path, {
                "idx": idx, "status": "ok", "elapsed_s": elapsed,
                "white": record["headers"].get("White"), "black": record["headers"].get("Black"),
                "event": record["headers"].get("Event"),
            })
            done += 1
            try:
                w = str(record["headers"].get("White"))
                b = str(record["headers"].get("Black"))
                print(f"[{idx:02d}] OK  {w} - {b}  ({elapsed}s)".encode("ascii", "replace").decode("ascii"))
            except Exception:
                pass  # never let a console-encoding issue mask a real success
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            tb = traceback.format_exc()
            _progress_line(progress_path, {
                "idx": idx, "status": "error", "elapsed_s": elapsed,
                "error": str(e), "traceback": tb[-4000:],
            })
            print(f"[{idx:02d}] FAIL {e} ({elapsed}s)")

    print(f"\nDone: {done}/{n_games} games successfully analyzed (attempted {attempted}).")


def summarize_only() -> None:
    diffs = []
    for fn in sorted(os.listdir(RAW_DIR)):
        if fn.endswith("_diff.json"):
            with open(os.path.join(RAW_DIR, fn), encoding="utf-8") as f:
                diffs.append(json.load(f))
    if not diffs:
        print("No diff files found in", RAW_DIR)
        return
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump([d["game_summary"] for d in diffs], f, indent=2)
    print(f"Wrote summary for {len(diffs)} games.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--llm-provider", type=str, default=None)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    if args.summarize_only:
        summarize_only()
    else:
        asyncio.run(run_batch(args.n_games, args.start, args.llm_provider))
        summarize_only()
