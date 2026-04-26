# BM25 Tantivy index (v2)

Build with:

```bash
python scripts/build_bm25_corpus.py --pgn-dir <path-to-annotated-pgns> --output data/bm25_positions_v2
```

## v2 schema (phase-aware RAG)

Indexed text fields (whitespace tokenizer): existing Bolčič-style fields plus `strategic_tags`, `endgame_signature`.

Stored raw fields include: `rag_phase` (`opening` | `middlegame` | `endgame`), `opening_eco`, `opening_prefix`, `opening_name`, `opening_matched_ply`, `opening_ply_bucket`, `material_signature`, `material_bucket`, `pawn_fingerprint`, `endgame_sig`, `corpus_version`, plus the previous `annotation_*`, `fen`, `pv_san`, `source`, etc.

`metadata.json` includes `corpus_version: "2"`, `encoder_version`, ply limits, and suggested RAG score keys.

Point the server at this directory:

```text
RAG_BM25_PATH=data/bm25_positions_v2
```

Tune retrieval strictness (optional; defaults reject weak matches):

- `RAG_MIN_SCORE_OPENING` (default `0.45`)
- `RAG_MIN_SCORE_MIDDLEGAME` (default `0.42`)
- `RAG_MIN_SCORE_ENDGAME` (default `0.40`)

Legacy indexes without `corpus_version: "2"` in `metadata.json` use the older retrieval path (no `rag_phase` filter).
