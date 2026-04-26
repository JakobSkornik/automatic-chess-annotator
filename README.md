# Automatic chess annotator (backend)

FastAPI service that analyzes a PGN with Stockfish, detects key moments, runs **BM25 (Tantivy)** retrieval over a local positional index, and generates concise LLM commentary. Results stream over the **jobs** WebSocket.

## Requirements

- Python 3.10+
- `stockfish.exe` in the project root (Windows) or `STOCKFISH_PATH` pointing at the binary
- OpenAI and/or Anthropic API key for commentary (choose provider per job; see env vars)

## Setup

```bash
python -m venv chess_v3.10
.\chess_v3.10\Scripts\activate.bat   # Windows
pip install -r requirements.txt
```

## Environment

| Variable | Purpose |
| -------- | ------- |
| `OPENAI_API_KEY` | Required when using provider `openai` |
| `ANTHROPIC_API_KEY` | Required when using provider `anthropic` |
| `LLM_DEFAULT_PROVIDER` | `openai` or `anthropic` (default `openai`) when a job omits `llm_provider` |
| `RAG_BM25_PATH` | Directory of the Tantivy BM25 index. **v2** phase-aware index default build: `data/bm25_positions_v2` (see below). Legacy: `data/bm25_positions`. If unset or invalid, commentary runs without reference examples |
| `RAG_BM25_STOCKFISH_DEPTH` | Depth for runtime PV used in BM25 queries (default `14`) |
| `RAG_MIN_SCORE_OPENING` | Minimum combined BM25+board score to inject opening examples (default `0.45`; below → no RAG) |
| `RAG_MIN_SCORE_MIDDLEGAME` | Same for middlegame (default `0.42`) |
| `RAG_MIN_SCORE_ENDGAME` | Same for endgame (default `0.40`) |
| `LLM_DEFAULT_EFFORT` | Optional default reasoning effort for the worker (`low` / `medium` / `high`). Concrete model ids per stage are fixed in `app/core/commentary/llm_policy.py`. |
| `LOG_LLM_PROMPTS` | Set to `1` or `true` to log full user/system prompts for each LLM pass (server `logger.info`; dev only) |

## Run

```bash
uvicorn app.main:app --reload
```

## API (jobs pipeline)

- `POST /jobs/submit` — submit PGN; returns `job_id`
- `GET /jobs/{job_id}/status` — job status
- `GET /jobs/{job_id}/game` — finished `GameJson` when ready
- `WebSocket /jobs/{job_id}/ws` — commentary and narrative updates

The legacy `/evaluator/*` interactive session API has been removed; use the jobs flow above.

## Architecture (pipeline)

```mermaid
flowchart LR
  submit[POST /jobs/submit] --> worker[analysis_worker]
  worker --> engine[Stockfish multipv + PV]
  engine --> evt[ChessEventExtractor]
  evt --> motifs[Tactical + strategic motifs]
  evt --> plans[Plan comparison]
  worker --> llm[GameAnnotationPipeline]
  llm --> fl[Future-line compare on critical moves]
  fl --> rat[MoveRationale JSON]
  rat --> cat[move_category]
  cat --> rag[BM25 Tantivy + board-sim rerank]
  rag --> prompts[Per-category composer]
  prompts --> mp[Tiered composer + quality audit]
  mp --> ws[WebSocket AI_COMMENT_UPDATE]
```

### Commentary pipeline (LLM)

1. **Engine + events** — `analyze_move` depth sweep (8/12/16), instability, PVs, motifs, `PlanComparison`, `move_category`.
2. **Critical moves** — optional dual `analyse` for **future-line delta** (played vs best PV leaves).
3. **RAG** — **v2 index** (`metadata.json` → `corpus_version: "2"`): BM25 with **phase Must** (`opening` / `middlegame` / `endgame` from the live position), phase-specific field boosts, `strategic_tags` / `endgame_signature` / ECO+ply buckets for opening, and **per-phase min score** env gates (weak match → no examples). **Legacy v1** indexes: same BM25 + board rerank without phase filter. Encoder still uses `positional_tokens` v2 fields.
4. **Prompting** — detected-motif glossary subset + `MoveRationale` JSON + tiered position block; **per-category** plain-text composer (`tactical`, `positional`, `defensive`, …) returning `named_motifs` + prose (then `auto_tokenize` for `[move:]`, `[pv:]`, etc.).
5. **Single composer call** — all key-moment tiers use one structured-output pass; **full** tiers get the full engine block, **compact/minimal** get trimmed rationale/RAG/position text and lower output caps (no narrator pre-pass).

## BM25 index

Build or refresh the Tantivy corpus from **annotated** PGNs (local one-off). The script walks `--pgn-dir` **recursively** for all `*.pgn` files. **Rebuild after encoder changes** (`encoder_version` in `positional_tokens.py`) or when changing phase features.

```bash
python scripts/build_bm25_corpus.py --help
# v2 default output directory:
#   data/bm25_positions_v2
# Defaults: --min-ply 1 --max-ply 200 (full game including endgames)
```

**Strict annotated mode (default):** each indexed position must have a non-junk `{...}` comment within the next `--max-annotation-delta` half-moves (default **4**). Positions with no such comment are **skipped** (smaller index than PV-only builds). Junk is filtered the same way as in `app/core/commentary/annotation_corpus.py` (too short, NAG-only, etc.).

**v2 rows** add `rag_phase`, ECO book fields, `endgame_sig` / endgame tag text, middlegame `strategic_tags`, `material_signature`, `pawn_fingerprint`, and `corpus_version`. At query time, [`TantivyPositionalRetriever`](app/core/commentary/tantivy_positional_retriever.py) prefers the master comment for RAG text and applies a `1/(1+Δ)` boost; if the comment refers to a position a few plies ahead, the UI may show the prefix `(annotation +N plies)`.

See [`data/bm25_positions_v2/README.md`](data/bm25_positions_v2/README.md) for the v2 schema summary. `metadata.json` records build parameters, `corpus_version`, and suggested RAG thresholds.

## Debugging RAG + prompts

**CLI — sample or query the BM25 index** (from repo root, with venv activated):

```bash
python scripts/random_rag_hits.py --count 5
python scripts/random_rag_hits.py --path data/bm25_positions_v2 --count 3
python scripts/random_rag_hits.py --query-fen "<FEN>" --pv-san "e5 Nf3 Nc6" --eco B12 --phase middlegame --ply 10 --top-k 5
```

Mode A prints random stored rows (source, game id, ECO, plies, FEN, PV SAN, annotation text). Mode B runs `TantivyPositionalRetriever.retrieve` and prints the same text the LLM sees (master comment or `PV:` fallback), score, and relevance tags.

**Server logs** — set `LOG_LLM_PROMPTS=1` to print the assembled event user blob (`build_event_input`) and each composer / JSON-schema pass with full system + user text.

**Frontend** — each streamed `AI_COMMENT_UPDATE` includes `data.llm_debug` (RAG query, rationale, planned system prompts, full user text, passes, token totals, timing, per-pass tokens, commentary audit). The web UI shows a collapsible **LLM debug** block under the active commentary card (below RAG chips) and logs the same payload to the browser console under `[LLM debug]`.

## Data flow

The backend is built around a single **jobs pipeline**: one HTTP submission creates one asynchronous job, one worker consumes it, and the same job id keys both the on-disk artifact and the commentary WebSocket. That design keeps the mental model simple: the client always correlates progress, streamed commentary, and the final JSON with one identifier.

Submission begins when the client posts a PGN to [`POST /jobs/submit`](app/routes/jobs.py). The route validates that the string contains exactly one game via [`PGNReader.validate_single_game`](app/core/io/pgn_reader.py), then calls [`queue_manager.add_job`](app/core/queue_manager.py). The queue manager assigns a UUID, stores the PGN plus optional `llm_provider` / `llm_effort`, enqueues the id on an asyncio queue, and records status `waiting`. The HTTP response is immediate; no analysis runs in the request thread.

The long-running work happens in [`analysis_worker`](app/core/worker.py), started from the FastAPI lifespan in [`app/main.py`](app/main.py). The worker dequeues a job id, loads job data, and sets status to `processing`. It then invokes [`run_engine_analysis_to_json`](app/core/engine/analysis_retriever.py), which parses the PGN, runs Stockfish-backed move analysis through [`EngineConnector`](app/core/engine/engine_connector.py), derives structured events and episodes (see **Modules** below), and returns an [`EnginePipelineState`](app/core/engine/analysis_retriever.py) together with enough material to build a [`GameJson`](app/models/GameJson.py). The worker writes that JSON to `data/games/{job_id}.json`—first **without** LLM fields—and advances status to `engine_complete`. That intermediate write lets a client or operator inspect engine output even if commentary fails.

Commentary is a second phase: [`GameAnnotationPipeline.run_llm_phases`](app/core/commentary/pipeline/game_pipeline.py) mutates the same `EnginePipelineState`, calling [`AdvancedCommentService`](app/core/commentary/advanced_comment_service.py) for digests, move prose, episode text, and a closing game narrative. Optional callbacks push WebSocket messages (see **LLM interactions**). When commentary finishes, the worker calls [`assemble_game_json`](app/core/engine/analysis_retriever.py) again and **overwrites** the same file with the enriched payload, clears the per-job commentary buffer, and sets status `completed`. Failures anywhere in the pipeline call [`mark_failed`](app/core/queue_manager.py) on the queue manager.

Real-time updates use [`WebSocket /jobs/{job_id}/ws`](app/routes/jobs.py). On connect, the server replays buffered messages so late subscribers do not miss earlier commentary. The worker’s `commentary_callback` both buffers and broadcasts via [`broadcast_to_job_ws`](app/core/queue_manager.py). Completed results are fetched with [`GET /jobs/{job_id}/game`](app/routes/jobs.py), which reads `data/games/{job_id}.json` when present. Separately, [`cleanup_old_files`](app/core/cleanup.py) runs in the background from the app lifespan and deletes stale files under `data/games` to bound disk use.

```mermaid
sequenceDiagram
  participant Client
  participant Routes as routes/jobs
  participant Queue as queue_manager
  participant Worker as worker
  participant Engine as EngineConnector
  participant AR as analysis_retriever
  participant LLM as AdvancedCommentService
  participant Disk as data/games
  Client->>Routes: POST /jobs/submit
  Routes->>Queue: add_job(pgn, llm opts)
  Queue-->>Client: JobResponse
  Worker->>Queue: dequeue job_id
  Worker->>AR: run_engine_analysis_to_json
  AR->>Engine: analyse per ply
  Worker->>Disk: write engine-only GameJson
  Worker->>Queue: status engine_complete
  Worker->>AR: GameAnnotationPipeline.run_llm_phases
  AR->>LLM: responses API plus RAG
  LLM-->>Queue: buffer + broadcast WS
  Worker->>Disk: overwrite with final GameJson
  Client->>Routes: GET /jobs/id/game
  Routes->>Disk: read JSON
```

## Modules

The codebase layers naturally from HTTP into orchestration, engine I/O, semantic extraction, retrieval, and language generation. The table below groups files by responsibility; follow the links when you need implementation detail.

| Layer | Role | Primary files |
| ----- | ---- | ------------- |
| API | FastAPI app, CORS, lifespan, health | [`app/main.py`](app/main.py) |
| HTTP + WebSocket | Job submit, status, game download, commentary socket | [`app/routes/jobs.py`](app/routes/jobs.py) |
| Queue + jobs | In-memory jobs, asyncio queue, WS fan-out, commentary replay | [`app/core/queue_manager.py`](app/core/queue_manager.py) |
| Worker | Dequeue loop, two-phase pipeline, file writes, status transitions | [`app/core/worker.py`](app/core/worker.py) |
| Retention | Periodic deletion of old game JSON | [`app/core/cleanup.py`](app/core/cleanup.py) |
| Engine | Stockfish subprocess via python-chess UCI | [`app/core/engine/engine_connector.py`](app/core/engine/engine_connector.py) |
| Pipeline | PGN → per-move analysis → events → episodes → `GameJson`; engine and LLM entry points | [`app/core/engine/analysis_retriever.py`](app/core/engine/analysis_retriever.py) |
| PGN I/O | Validate and read a single game from a string | [`app/core/io/pgn_reader.py`](app/core/io/pgn_reader.py) |
| Events | From analyzed rows to `MoveEvent` (motifs, plans, ECO, key moments) | [`app/core/commentary/event_extractor.py`](app/core/commentary/event_extractor.py) |
| Episodes | Segment event streams into narrative chunks | [`app/core/commentary/episode_segmenter.py`](app/core/commentary/episode_segmenter.py) |
| Key moments | Heuristic labels that gate LLM move commentary | [`app/core/commentary/key_moment_detector.py`](app/core/commentary/key_moment_detector.py) |
| Rationale | Deterministic JSON summarizing a move for prompts | [`app/core/commentary/move_rationale.py`](app/core/commentary/move_rationale.py) |
| Features | Classification, motifs, plan comparison, future-line compare, positional features and token encoders | [`app/core/commentary/features/`](app/core/commentary/features) |
| Openings | ECO lookup from bundled JSON | [`app/core/commentary/openings/eco_book.py`](app/core/commentary/openings/eco_book.py) |
| RAG | Query model and default retriever wiring | [`app/core/commentary/rag_retriever.py`](app/core/commentary/rag_retriever.py), [`app/core/commentary/tantivy_positional_retriever.py`](app/core/commentary/tantivy_positional_retriever.py) |
| Corpus helpers | Offline PGN comment filtering (index build scripts; not on the hot path) | [`app/core/commentary/annotation_corpus.py`](app/core/commentary/annotation_corpus.py) |
| LLM + tokens | OpenAI Responses API, prompts, post-processing into `[type:content]` tokens | [`app/core/commentary/advanced_comment_service.py`](app/core/commentary/advanced_comment_service.py), [`app/core/commentary/annotation_tokens.py`](app/core/commentary/annotation_tokens.py) |
| Models | API and domain types | [`app/models/GameJson.py`](app/models/GameJson.py), [`app/models/job.py`](app/models/job.py), [`app/models/Move.py`](app/models/Move.py), [`app/models/PgnMetadata.py`](app/models/PgnMetadata.py), [`app/models/chess_events.py`](app/models/chess_events.py) |
| Config | Small static `Settings` holder | [`app/config/settings.py`](app/config/settings.py) |
| Env bootstrap | Optional `dotenv`, exposes `OPENAI_API_KEY` | [`app/core/__init__.py`](app/core/__init__.py) |

## LLM interactions

All generative language in this service goes through **one module**: [`AdvancedCommentService`](app/core/commentary/advanced_comment_service.py). It constructs prompts, calls **`AsyncOpenAI().responses.create`** (OpenAI’s **Responses** API, not Chat Completions), and parses structured outputs where JSON schemas are attached. There is no second LLM SDK in `app/`. Environment variables for models, logging, and RAG paths are summarized in **Environment** above; additionally, the commentary stage respects **`FUTURE_LINE_DEPTH`** and **`FUTURE_LINE_PLIES`** when comparing played lines to the engine’s best continuation (see [`future_line_compare.py`](app/core/commentary/features/future_line_compare.py)).

Orchestration lives in [`GameAnnotationPipeline`](app/core/commentary/pipeline/game_pipeline.py). Conceptually it runs four **language** passes over the same `GameAnalysisContext` carried in `EnginePipelineState`, interleaved with **non-LLM** Stockfish calls that only exist to sharpen retrieval and rationale (principal variation for BM25, future-line deltas on critical positions).

First, **`generate_game_digest`** asks the model for a compact structured summary (including **`strategic_archetype`** and **`turning_points[].motif`**). The digest is stored on the context and copied to **`GameMetadata.strategic_archetype`** when present; **`GAME_SUMMARY`** streams to clients when configured. Second, for moves with a **`key_moment_type`** or a per-episode **teaching** highlight, the pipeline may refresh PV SAN for BM25 and optionally a **`FutureLineDelta`** when the played move diverges from the engine’s preference, then **`analyze_and_compose_event`** runs. Prompts inject digest context, archetype idea bank, rolling **`prior_context_snippets`**, and **DETECTED_MOTIFS**; full engine dumps apply only on the highest tier. Output is structured segment JSON (with **`named_motifs`**) assembled to inline tokens, then **`auto_tokenize`**. **`llm_debug`** includes **`elapsed_ms`**, **`tokens_by_pass`**, and **`commentary_audit`** (motif coverage, eval-token count, forbidden-phrase hits). Back-to-back duplicate key-moment labels within two plies collapse to a short stub reference instead of a second LLM call.

Third, **`generate_episode_commentary`** runs in parallel across episodes (after move commentary), producing **`EPISODE_NARRATIVE`** messages. Fourth, **`generate_game_narrative`** closes the game using result, Elos, digest turning points, and episode summaries (**`GAME_NARRATIVE`**).

Retrieval is **lexical**, not embedding-based. [`build_rag_query`](app/core/commentary/rag_retriever.py) packs FEN, PV SAN, ECO, phase, motifs, and theme hints into a [`RAGQuery`](app/core/commentary/rag_retriever.py). [`get_default_retriever`](app/core/commentary/tantivy_positional_retriever.py) opens a Tantivy index at **`RAG_BM25_PATH`** or returns a no-op retriever if the path is missing. Ranking blends normalized BM25 with board-structure similarity and boosts examples whose master annotation is closer in plies (see **Architecture** and **BM25 index** for the scoring intuition). If **`OPENAI_API_KEY`** is absent, commentary degrades gracefully: digests may be empty and move commentary falls back to a fixed disabled message, while engine analysis still completes.

```mermaid
flowchart TD
  digest[generate_game_digest]
  subgraph keyMoves [Keyed moves only]
    pv[Stockfish PV for BM25]
    fl[Future-line compare]
    rag[Tantivy retrieve]
    comp[analyze_and_compose_event]
    tok[auto_tokenize]
    pv --> rag
    fl --> comp
    rag --> comp
    comp --> tok
  end
  ep[generate_episode_commentary per Episode]
  gn[generate_game_narrative]
  digest --> keyMoves
  keyMoves --> ep
  ep --> gn
```

## PGN input format

- **Required:** a single game that `python-chess` can parse (see [`PGNReader.validate_single_game`](app/core/io/pgn_reader.py)).
- **Optional `[ECO "XYY"]`:** use standard codes **`A00`–`E99`** only (three characters, case-sensitive, no spaces). Malformed tags are ignored with a warning; internal book detection still runs.
- **Optional `[Opening "..."]`:** Lichess/Chess.com-style names help match a variation when they align with detected theory. Values like empty, `?`, `Unknown`, or `Custom` are treated as absent and the internal ECO/longest-prefix logic supplies opening metadata.

Example header block:

```pgn
[Event "Casual"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]
[ECO "C45"]
[Opening "Scotch Game"]
```

See also **[`docs/motif_glossary.md`](docs/motif_glossary.md)** for motif enum values and coach phrasings.

## Tests

```bash
python -m pytest tests -q
```
