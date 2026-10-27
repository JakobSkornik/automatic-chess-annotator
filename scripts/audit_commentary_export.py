"""Audit a benchmark worker's saved artifacts without engine or LLM calls.

Checks legal replay, evidence anchors, selection provenance and exact approved
prose. These checks are not a semantic correctness or human usefulness score.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import chess.pgn


def summarize(game):
    counts, selection_counts = Counter(), Counter()
    rows = []
    for ply, move in enumerate(game["moves"], 1):
        debug = move.get("debug") or {}
        validation = debug.get("commentary_validation") or {}
        facts = move.get("comment_facts") or {}
        selection = facts.get("selection_metadata", {}).get("comment_selection", {})
        if selection:
            selection_counts[selection["reason"]] += 1
        if debug.get("rendering_contract"):
            counts[validation.get("semantic_review", "not_recorded")] += 1
        counts["substantive_comments"] += bool(move.get("final_comment"))
        if selection.get("allowed") is False:
            assert not move.get("final_comment"), f"Blocked move has substantive prose: {ply}"
        if move.get("final_comment") or debug.get("rendering_contract"):
            rows.append({"ply": ply, "san": move["san"], "text": move.get("comment"),
                         "selection": selection, "review": validation.get("semantic_review"),
                         "fallback_reason": validation.get("fallback_reason"),
                         "words": len((move.get("comment") or "").split()),
                         "fact_count": len(facts.get("claims", [])) + len(
                             (facts.get("better_alternative") or {}).get("claims", []))})
    return {"plies": len(game["moves"]), "counts": dict(counts),
            "selection_counts": dict(selection_counts), "annotations": rows}


def validate(folder, data):
    request = json.loads((folder / "request.json").read_text(encoding="utf-8-sig"))
    original = chess.pgn.read_game(io.StringIO(request["case"]["pgn"]))
    exported = chess.pgn.read_game(io.StringIO((folder / "game.pgn").read_text(encoding="utf-8")))
    assert original and exported and not original.errors and not exported.errors
    assert list(original.mainline_moves()) == list(exported.mainline_moves())
    assert original.headers["Result"] == exported.headers["Result"]
    assert data["commentary_complete"] is True
    assert len(data["moves"]) == len(list(original.mainline_moves()))
    board, node = original.board(), exported
    counts = Counter()
    prose_mismatches = []
    for ply, (expected, move) in enumerate(zip(original.mainline_moves(), data["moves"]), 1):
        origin = board.fen()
        assert expected.uci() == move["uci"] and board.san(expected) == move["san"]
        board.push(expected)
        assert board.fen() == move["fen"]
        counts["mainline_positions"] += 1
        facts = move.get("comment_facts") or {}
        debug = move.get("debug") or {}
        validation = debug.get("commentary_validation") or {}
        for branch in (facts, facts.get("better_alternative") or {}):
            line = branch.get("display_line")
            if not line:
                continue
            replay, pv_node = chess.Board(origin), node
            assert line["start_fen"] == origin
            assert len(line["san"]) == len(line["fens"])
            check_export = move.get("final_comment") and debug.get("rendering_contract") == "grounded_narrative_v3"
            tail_in_comment = False
            for offset, (san, fen) in enumerate(zip(line["san"], line["fens"])):
                played = replay.parse_san(san)
                replay.push(played)
                assert replay.fen() == fen
                if check_export and not tail_in_comment:
                    if not pv_node.has_variation(played) and pv_node is exported.end():
                        # Analysis after the last recorded move must not extend
                        # the game's mainline; the exporter stores this as text.
                        tail = "Continuation: " + " ".join(line["san"][offset:]) + "."
                        assert tail in pv_node.comment, f"Missing terminal PV at ply {ply}"
                        tail_in_comment = True
                    else:
                        assert pv_node.has_variation(played), f"Missing exported PV at ply {ply}: {line['san']}"
                        pv_node = pv_node.variation(played)
            assert replay.fen() == line["leaf_fen"]
            counts["replayed_branches"] += 1
            counts["exported_narrative_branches"] += bool(check_export)
            for claim in branch.get("claims", []):
                anchor = claim["support_ply"]
                assert type(anchor) is int and 1 <= anchor <= len(line["fens"])
                assert claim["baseline_fen"] == origin
                assert claim["support_fen"] == line["fens"][anchor - 1]
                assert claim["realization"] == ("immediate" if anchor == 1 else "envisioned")
                counts["consistent_claim_anchors"] += 1
        if debug.get("rendering_contract") == "grounded_narrative_v3" and validation.get("semantic_review") == "accepted":
            catalog, selected = validation["claim_catalog"], validation["selected_claims"]
            approved = selected["text"]
            for key, value in catalog["substitutions"].items():
                approved = approved.replace("{{" + key + "}}", value)
            if move["comment"] != approved.strip():
                prose_mismatches.append({"ply": ply, "approved": approved.strip(), "exported": move["comment"]})
            else:
                counts["unchanged_approved_annotations"] += 1
        node = node.variation(expected)
    return {"legal_replay_and_anchors_passed": True, "approved_prose_unchanged": not prose_mismatches,
            "prose_mismatches": prose_mismatches, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    game = json.loads((args.folder / "game.json").read_text(encoding="utf-8"))
    result = {"scope": "Single-game development verification, not a controlled quality benchmark.",
              "candidate": summarize(game), "integrity": validate(args.folder, game),
              "runtime": json.loads((args.folder / "runtime.json").read_text(encoding="utf-8"))}
    paths = [args.folder / name for name in ("request.json", "game.json", "game.pgn", "runtime.json")]
    if args.baseline:
        result["baseline"] = summarize(json.loads(args.baseline.read_text(encoding="utf-8-sig")))
        paths.append(args.baseline)
    result["sha256"] = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (args.folder / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity = {key: value for key, value in result["integrity"].items() if key != "prose_mismatches"}
    integrity["prose_mismatch_count"] = len(result["integrity"]["prose_mismatches"])
    print(json.dumps({"candidate": result["candidate"]["counts"], "integrity": integrity}))


if __name__ == "__main__":
    main()
