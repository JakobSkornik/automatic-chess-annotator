"""Offline smoke test: run the engine pipeline over one PGN and verify the
FE-facing GameJson contract is intact (all keys present, additive-only)."""

import asyncio
import json
import sys

sys.path.insert(0, ".")

import chess.pgn
import io


def first_game_string(path: str) -> str:
    games = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            out = io.StringIO()
            print(g, file=out)
            games.append(out.getvalue())
            if len(games) >= 1:
                break
    return games[0] if games else ""


async def main() -> int:
    import glob

    candidates = sorted(glob.glob("data/pgn/manual/*.pgn"))
    pgn1 = ""
    for c in candidates:
        try:
            pgn1 = first_game_string(c)
        except Exception:
            continue
        if pgn1.strip():
            break
    if not pgn1:
        # Fall back to a synthetic game so the contract check still runs.
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        for mv in [
            "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6",
            "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6", "c2c3", "e8g8",
            "h2h3", "c6a5", "b3c2", "c7c5", "d2d4",
        ]:
            node = node.add_variation(chess.Move.from_uci(mv))
            board.push(chess.Move.from_uci(mv))
        buf = io.StringIO()
        print(game, file=buf)
        pgn1 = buf.getvalue()

    from app.core.engine.analysis_retriever import run_engine_analysis_to_json
    from app.core.io.pgn_reader import PGNReader

    PGNReader.validate_single_game(pgn1)
    from app.core.engine.engine_connector import get_global_engine_connector

    conn = get_global_engine_connector()

    async def cb(p, m):
        pass

    gj, state = await run_engine_analysis_to_json(
        pgn1, conn, cb, metadata_id="smoke-test"
    )
    d = json.loads(gj.model_dump_json())
    moves = d["moves"]
    commented = [m for m in moves if m.get("comment")]
    print("moves:", len(moves), "| commented:", len(commented))
    required = [
        "mn", "color", "san", "uci", "fen", "phase", "score", "variations",
        "comment", "classification", "annotation", "is_key_moment",
        "final_comment", "resolved_tokens", "comment_facts", "debug",
    ]
    missing = [(i, k) for i, m in enumerate(moves) for k in required if k not in m]
    print("missing FE keys:", missing[:5] or "none")
    cf = next((m["comment_facts"] for m in commented if m.get("comment_facts")), None)
    if cf:
        print("comment_facts keys:", sorted(cf.keys()))
        if cf.get("claims"):
            print("claims[0] keys:", sorted(cf["claims"][0].keys()))
        dl = cf.get("display_line") or {}
        print("display_line keys:", sorted(dl.keys()))
        print("feature_series present:", bool(dl.get("feature_series")))
    dbg = next((m["debug"] for m in commented if m.get("debug")), None)
    print("debug has refutations field:", bool(dbg) and "refutations" in dbg)
    print("top-level keys:", sorted(d.keys()))
    ai = d.get("analysis_info") or {}
    print("analysis_info keys:", sorted(ai.keys()) if isinstance(ai, dict) else ai)
    print("metadata keys:", sorted((d.get("metadata") or {}).keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
