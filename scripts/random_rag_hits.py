#!/usr/bin/env python3
"""
Inspect the Tantivy BM25 corpus: random rows (Mode A) or a retrieval query (Mode B).
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Tuple

import tantivy
from tantivy import Occur, Query

# Repo root (parent of scripts/) — so `python scripts/random_rag_hits.py` finds `app`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "bm25_positions"


def _wrap(text: str, width: int = 88) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return "\n".join(textwrap.wrap(t, width=width))


def _all_doc_addresses(index: tantivy.Index) -> Tuple[List[Any], int]:
    """Return (addresses, num_docs) using a boolean Should over player_color w|b."""
    index.reload()
    searcher = index.searcher()
    n = int(searcher.num_docs)
    if n == 0:
        return [], 0
    q_w = index.parse_query("w", default_field_names=["player_color"])
    q_b = index.parse_query("b", default_field_names=["player_color"])
    bool_q = Query.boolean_query([(Occur.Should, q_w), (Occur.Should, q_b)])
    result = searcher.search(bool_q, limit=max(n, 1))
    addrs = [addr for _score, addr in result.hits]
    return addrs, n


def _print_doc_block(searcher: Any, addr: Any, idx: int) -> None:
    doc = searcher.doc(addr)
    source = (doc.get_first("source") or "").strip()
    game_id = (doc.get_first("game_id") or "").strip()
    eco = (doc.get_first("eco") or "").strip()
    plies = (doc.get_first("plies_to_next_annotation") or "").strip()
    ann_ply = (doc.get_first("annotation_ply") or "").strip()
    fen = (doc.get_first("fen") or "").strip()
    pv_san = (doc.get_first("pv_san") or "").strip()
    ann = (doc.get_first("annotation_text") or "").strip()
    print(f"--- sample {idx} ---")
    print(f"source: {source}")
    print(f"game_id: {game_id}")
    print(f"eco: {eco}")
    print(f"plies_to_next_annotation: {plies}")
    print(f"annotation_ply: {ann_ply}")
    print(f"fen: {fen}")
    print(f"pv_san: {pv_san}")
    print("annotation_text:")
    wrapped = _wrap(ann, width=88)
    print(wrapped if wrapped else "(empty)")
    print()


def mode_a_random(path: Path, count: int) -> None:
    if not path.is_dir():
        print(f"Index path not found: {path}", file=sys.stderr)
        sys.exit(1)
    index = tantivy.Index.open(str(path))
    addrs, n = _all_doc_addresses(index)
    if n == 0:
        print("Corpus has 0 documents.")
        return
    k = min(count, len(addrs))
    sample = random.sample(addrs, k=k)
    index.reload()
    searcher = index.searcher()
    nonempty = 0
    for addr in addrs:
        doc = searcher.doc(addr)
        ann = (doc.get_first("annotation_text") or "").strip()
        if ann:
            nonempty += 1
    for i, addr in enumerate(sample, 1):
        _print_doc_block(searcher, addr, i)
    print(
        f"Totals: {n} rows in corpus, {nonempty} have non-empty annotation_text"
    )


async def mode_b_query(
    path: Path,
    fen: str,
    pv_san: List[str],
    eco: str | None,
    phase: str | None,
    ply: int | None,
    tactical_motifs: List[str],
    top_k: int,
) -> None:
    from app.core.commentary.rag_retriever import RAGQuery
    from app.core.commentary.tantivy_positional_retriever import TantivyPositionalRetriever

    q = RAGQuery(
        fen=fen,
        pv_san=pv_san,
        eco=eco,
        phase=phase,
        ply=ply,
        tactical_motifs=tactical_motifs,
    )
    retriever = TantivyPositionalRetriever(str(path))
    results = await retriever.retrieve(q, top_k=top_k)
    if not results:
        print("No hits (check FEN/PV and index path).")
        return
    for i, r in enumerate(results, 1):
        print(f"--- hit {i} ---")
        print(f"similarity_score: {r.similarity_score}")
        print(f"relevance_tags: {r.relevance_tags}")
        print("LLM text (annotation or PV fallback):")
        print(_wrap(r.annotation_text, width=88))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample or query the BM25 Tantivy index.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help=f"Tantivy index directory (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument("--count", type=int, default=5, help="Random samples (Mode A).")
    parser.add_argument("--query-fen", type=str, default=None, help="Mode B: FEN after the move.")
    parser.add_argument(
        "--pv-san",
        type=str,
        default=None,
        help='Mode B: space-separated SAN continuation, e.g. "e5 Nf3 Nc6"',
    )
    parser.add_argument("--eco", type=str, default=None, help="Mode B: ECO code.")
    parser.add_argument("--phase", type=str, default=None, help="Mode B: game phase.")
    parser.add_argument("--ply", type=int, default=None, help="Mode B: ply.")
    parser.add_argument(
        "--tactical-motifs",
        type=str,
        default="",
        help='Mode B: comma-separated tactical motif tags (optional).',
    )
    parser.add_argument("--top-k", type=int, default=5, help="Mode B: number of hits.")
    args = parser.parse_args()
    path = args.path.resolve()

    if args.query_fen is not None:
        if not args.pv_san:
            print("Mode B requires --pv-san.", file=sys.stderr)
            sys.exit(2)
        pv_list = [x for x in args.pv_san.split() if x]
        motifs = [x.strip() for x in args.tactical_motifs.split(",") if x.strip()]
        asyncio.run(
            mode_b_query(
                path,
                args.query_fen,
                pv_list,
                args.eco,
                args.phase,
                args.ply,
                motifs,
                args.top_k,
            )
        )
    else:
        mode_a_random(path, args.count)


if __name__ == "__main__":
    main()
