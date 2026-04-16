#!/usr/bin/env python3
"""Smoke-test the RAG corpus: embed a query and print Chroma nearest neighbors.

Uses the same model and collection name as scripts/build_rag_corpus.py.

Example (PowerShell):
  $env:OPENAI_API_KEY="..."
  $env:PYTHONPATH="c:\\School\\automatic-chess-annotator"
  python scripts/query_rag_corpus.py --query "d4 in opening, closed center. space advantage"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "chess_annotations"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "rag_corpus",
        help="ChromaDB persist path (same as --output-dir for build_rag_corpus.py)",
    )
    p.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural language query (ideally shaped like ingest text: move + phase + center + idea)",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=("opening", "middlegame", "endgame"),
        help="Optional metadata filter on stored phase",
    )
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to embed the query.", file=sys.stderr)
        sys.exit(1)

    corpus_dir = args.corpus_dir.resolve()
    if not corpus_dir.is_dir():
        print(f"Corpus dir not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    import chromadb
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[args.query])
    q_emb = resp.data[0].embedding

    ch = chromadb.PersistentClient(path=str(corpus_dir))
    coll = ch.get_collection(COLLECTION_NAME)

    where = {"phase": args.phase} if args.phase else None
    result = coll.query(
        query_embeddings=[q_emb],
        n_results=args.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = result.get("ids") or [[]]
    docs = result.get("documents") or [[]]
    metas = result.get("metadatas") or [[]]
    dists = result.get("distances") or [[]]

    print(f"collection={COLLECTION_NAME} path={corpus_dir}\nquery={args.query!r}\n")
    for i in range(len(ids[0])):
        print(f"--- rank {i + 1} distance={dists[0][i]!r} ---")
        print(f"id: {ids[0][i]}")
        m = metas[0][i] or {}
        print(f"phase={m.get('phase')} pawn_structure={m.get('pawn_structure_type')} san={m.get('san')}")
        doc = docs[0][i] or ""
        print(doc[:800] + ("..." if len(doc) > 800 else ""))
        print()


if __name__ == "__main__":
    main()
