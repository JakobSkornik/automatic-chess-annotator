"""Debug: replicate the eval_harness path EXACTLY to find the PV SAN bug."""

import asyncio
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import chess


async def main() -> None:
    import eval_harness as eh

    from app.core.commentary.event_extractor import ChessEventExtractor
    from app.core.commentary.features.envisioned import (
        trim_envisioned_line_by_probe,
    )
    from app.core.commentary.openings.eco_book import ECOBook
    from app.core.commentary.phase_classifier import PhaseClassifier
    from app.core.commentary.rules import build_comment_facts
    from app.core.engine.analysis.constants import DEFAULT_ANALYSIS_DEPTH
    from app.core.engine.analysis.engine_pass import AnalysisRetriever
    from app.core.engine.engine_connector import EngineConnector, stockfish_path
    from app.core.io.pgn_reader import PGNReader
    from app.models.chess_events import AnalyzedMoveData
    from app.models.Move import Move

    conn = EngineConnector(stockfish_path)
    wrapped = eh.CountingEngineWrapper(conn)

    games = eh._iter_pgn_games(eh.os.path.abspath("data/pgn/GM_games.pgn"))
    print(f"harness splitter: {len(games)} games; chunk0 head:")
    print("\n".join(games[0].splitlines()[:8]))

    game = PGNReader().read_game_from_string(games[0])
    print(
        "game:",
        game.headers.get("White"),
        "vs",
        game.headers.get("Black"),
        "| plies:",
        len(list(game.mainline_moves())),
    )

    eco_book = ECOBook()
    retriever = AnalysisRetriever(wrapped, game, eco_book)
    pc = PhaseClassifier(eco_book)
    rows: list[AnalyzedMoveData] = []
    for idx, mo in enumerate(retriever.get_move_list()):
        ba = chess.Board(mo.position)
        mo.phase = pc.classify(ba, retriever._uci_prefix_for_depth(mo.depth))
        if mo.phase == "early":
            analyzed, pvs = retriever.analyze_book_move(mo)
        else:
            analyzed, pvs = retriever.analyze_move(mo, stage=DEFAULT_ANALYSIS_DEPTH)
        em = (
            analyzed.hiddenFeatures.get("_engine", {})
            if isinstance(analyzed.hiddenFeatures, dict)
            else {}
        )
        bb = game.board()
        for i, gm in enumerate(game.mainline_moves()):
            if i >= idx:
                break
            bb.push(gm)
        rows.append(
            AnalyzedMoveData(
                index=idx,
                ply=mo.depth,
                san=bb.san(chess.Move.from_uci(mo.move)),
                uci=mo.move,
                fen_before=bb.fen(),
                fen_after=mo.position,
                score_cp=analyzed.score,
                phase_raw=str(mo.phase),
                pvs=pvs or [],
                hidden_features=analyzed.hiddenFeatures or {},
                analyzed_move=analyzed,
                eval_at_depth=dict(em.get("eval_at_depth") or {}),
                pv1_change_count=int(em.get("pv1_change_count", 0)),
            )
        )

    extractor = ChessEventExtractor(
        eco_book=eco_book, key_moment_detector=retriever.key_moment_detector
    )
    events = extractor.extract_events(game, rows)

    def prober(fen: str):
        return wrapped.evaluate_position(fen, depth=6)

    n_fail = 0
    for mi, me in enumerate(events):
        if me.phase == "opening" or not (0 <= me.move_index < len(rows)):
            continue
        row = rows[me.move_index]
        facts = build_comment_facts(row, me, depth=DEFAULT_ANALYSIS_DEPTH)
        if facts is None or facts.display_line is None or not facts.claims:
            continue
        dl = facts.display_line
        b = chess.Board(dl.start_fen)
        ok = True
        for i, san in enumerate(dl.line_san):
            try:
                b.push(b.parse_san(san))
            except Exception as e:
                n_fail += 1
                emeta = (row.hidden_features or {}).get("_engine") or {}
                print(f"PLY {me.ply} FAIL tok {i} '{san}': {e}")
                print("  played:", me.san, me.uci, "| fen_before:", dl.start_fen[:80])
                print("  after_pv:", emeta.get("after_pv_uci"))
                print("  line_uci:", dl.line_uci)
                print("  line_san:", dl.line_san)
                ok = False
                break
        if n_fail >= 2:
            break
    print(f"done; failures={n_fail}", flush=True)
    conn.close()
    import os

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    asyncio.run(main())
