# Refactor Plan v2 — Merging Guid's Annotator Architecture with LLM Enrichment

Supersedes v1 (pure de-humanization). Source: Matej Guid's dissertation, Ch. 5 (pp. 69–83),
feature material from Ch. 6. Scope: backend (`automatic-chess-annotator`); frontend items are
specified only as data contracts.

## Implementation status (2026-06-11, branch `guid-refactor`)

All backend steps (0–9) are implemented and tested (123 tests pass; offline smoke test on a
52-ply game incl. endgame). New modules: `commentary/phase_classifier.py`,
`commentary/phases/{early,composer}.py`, `commentary/features/{guid_features,envisioned}.py`,
`commentary/rules/engine.py`, `models/comment_facts.py`, `io/pgn_writer.py`,
endpoint `GET /jobs/{id}/pgn`.

Env knobs: `HUMANIZATION_LEVEL` (0 template / 1 polish / 2 flavor; default 2),
`ENDGAME_PIECE_THRESHOLD` (7), `ENVISIONED_MAX_PLIES` (11), `BOOK_EXIT_SEED_DEPTH` (16),
`CLAIM_DEDUP_WINDOW_PLIES` (6), `PV_HORIZON_ENABLED` (0).

Deviations from the plan below:
- **No `[pv:#k]` by-reference tokens.** Comments keep inline `[pv:...]`; the fact contract
  forces the LLM to copy them verbatim (validated post-call, template fallback otherwise), and
  `GameMove.resolved_tokens` + `Variation.fens` give the popup per-step san/fen/from/to anyway.
- `pv_horizon_diff` is disabled by default (env flag) rather than deleted; `delta_to_motif`
  still consumes it when re-enabled. `future_line_compare` runs only for moves without
  CommentFacts (legacy path).
- `generate_game_digest`/`build_game_digest_input` remain in `advanced_comment_service.py` as
  dead code — cleanup candidate.
- Adjacent-move claim repetition is handled by a forward dedup window at facts-build time, not
  in the reverse pass.

## 0. Goals

1. **Guid's structure, our voice.** Rebuild the backend around Guid's three modules
   (Search / Knowledge / Expert) so every comment is grounded in a feature-difference vector and
   fired rules — but keep an LLM layer on top that adds flavor, opening metadata (RAG), and
   game-level context. Humanization stays, hallucination goes: the LLM may decorate facts, not
   invent them.
2. **Phase-split generation.** Early game = ECO-book hits → **no engine at all**; late game =
   standard material criterion; middle game = the rest. Different comment strategies per phase.
3. **Reverse-order commenting.** Generate comments from the last move backwards so the LLM knows
   how the game ends and can foreshadow ("this weak dark-square complex decides the endgame").
4. **Feature-progression charts.** Per-ply time series for every positional feature + per-comment
   references to the features that mattered, so the UI can render a grid of charts and highlight
   the relevant ones for the selected comment.
5. **Scrap the game summary** (`game_summary` + the digest LLM stage).
6. **PGN export** with comments, eval, PV (as variations), and optionally fired features.
7. **Interactive comments**: keep the `[type:content]` token system; PVs become one popup per
   line with slider / autoplay / step controls — backend supplies fully resolved lines.

---

## 1. Where we are (findings recap, validated against code + generated JSONs)

- Engine (Stockfish, NNUE off) runs **for every ply including book moves**: depth-16 eval,
  depth-8/12 probes, multipv-3 PVs, pv-horizon — ~6 searches per ply
  (`app/core/engine/analysis_retriever.py:938-996`). Book moves then get **no comment anyway**.
- Phase (`"early"` = full ECO-prefix hit, `"end"` = minor+major pieces < 7, else `"mid"`) is
  already computed (`analysis_retriever.py:640-677`) but gates nothing and is **absent from the
  output JSON**.
- Comments are freeform LLM prose; no rule layer. The style guide
  (`advanced_comment_service.py:555-570`) bans numeric evals ("use words") → *"about a
  seven-pawn edge"*. Positional-feature claims appear in only ~13 % of comments; coverage ~58 %.
- Guid's diff-vector idea exists twice in embryo, both unused downstream:
  `features/pv_horizon_diff.py` (audit-only) and `features/future_line_compare.py`
  (key moments, prose hint only).
- `engine.trace()` is **dead code** (python-chess `SimpleEngine` has no `trace()`; always raises,
  swallowed at `analysis_retriever.py:314-317`).
- Token grammar already exists (`annotation_tokens.py`): types `pv, move, square, file, eval,
  piece`; `[pv:...]` resolves to per-step `{san, fen, from, to}` via `_resolve_pv_line`.
  **But** `resolved_tokens` (built at `game_pipeline.py:251-261`) never reaches the final
  `GameMove` JSON, and `variations[].line` is SAN-only — no FENs. The PV popup needs both fixed.
- `game_summary` = the digest dict (`strategic_archetype`, `turning_points`, `phase_story`, …);
  the archetype and turning points are also injected into move prompts
  (`game_pipeline.py:68-76,226`), so scrapping it needs a replacement for those inputs.
- Current per-move LLM calls are parallelized per episode; reverse-order generation changes this
  (see §5).

---

## 2. Target architecture

```
                      ┌────────────────────────────────────────────────┐
 PGN ──► PhaseClassifier (ECO book, material count — no engine)        │
          │ early           │ mid / end                                │
          ▼                 ▼                                          │
   (no engine)        SEARCH MODULE                                    │
   opening meta       depth-16 multipv-3, book-exit seed eval,         │
   from ECO + RAG     shortened+quiescent PVs                          │
          │                 │                                          │
          ▼                 ▼                                          │
        KNOWLEDGE MODULE  (all plies, engine-free, pure python)        │
        weighted feature vector per position; per-ply feature_series   │
                            │                                          │
                            ▼                                          │
        EXPERT MODULE  (mid/end only)                                  │
        diff vectors start→envisioned, played-vs-best;                 │
        rules fire → CommentFacts {verdict, line, eval, claims,        │
                                   features_involved}                  │
                            │                                          │
                            ▼  (iterating LAST move → FIRST)           │
        LLM ENRICHMENT LAYER                                           │
        facts inviolable + flavor, RAG opening metadata,               │
        future-context buffer (later comments, result, episode arc)    │
                            │                                          │
                            ▼                                          │
        OUTPUT CONTRACTS: GameJson (+phase, +feature_series,           │
        +feature_refs, +resolved variations) │ PGN export              │
                      └────────────────────────────────────────────────┘
```

The split of responsibilities is the heart of the merge:

| Layer | Owns | May NOT do |
|---|---|---|
| Expert Module (rules) | every positional **claim**, the verdict, the displayed line, the eval, when to stay silent | prose style |
| LLM layer | voice, flavor, transitions, opening lore (from supplied RAG snippets), foreshadowing (from supplied future context), interactive tokens | introduce any positional claim, eval, or line not present in its input |

### 2.1 Phase gating (Search Module)

- New `PhaseClassifier` running **before** any engine call (needs only board + UCI prefix +
  `ECOBook.match`). Definitions: `early` while the full prefix matches the ECO book; `end` when
  minor+major pieces < threshold (`ENDGAME_PIECE_THRESHOLD`, default 7 as today, configurable —
  confirm convention with Matej); `mid` otherwise.
- `run_engine_analysis_to_json` loop: `early` plies skip `analyze_move` entirely — lightweight
  row with san/uci/fens/phase/opening info, `score=None`. One **seed eval** of the last book
  position so the first novelty has a `prevScore` for swing detection; guard
  `key_moment_detector`/event extractor against `score=None`.
- `mid`/`end`: engine as today **plus** envisioned-position machinery (§2.3). Remove the dead
  `trace()` plumbing and the audit-only `pv_horizon_diff` while in there.

### 2.2 Knowledge Module — `features/guid_features.py`

- `FeatureValue {name, side, value_cp, flag: Optional[int]}`;
  `compute_feature_vector(board) -> dict[str, FeatureValue]`.
- Computed for **every mainline ply, all phases** (it's engine-free and cheap) — this is what
  feeds the chart widget — and additionally for envisioned positions and PV nodes in mid/end.
- Feature inventory: §7 table (Crafty set from Ch. 5 + Guid's piece-activity additions +
  Ch. 6 `BAD_BISHOP`, folding in the useful existing features from `positional_features.py`
  with centipawn weights). All weights/thresholds live in one config (Guid §5.4.2: manual
  tunability is a design requirement).

### 2.3 Expert Module — diff vectors + rules → `CommentFacts`

- **Envisioned position**: engine PV shortened from the tail while non-quiescent (pending
  captures/checks/recaptures), capped by audience config (~10–12 plies). Starting position must
  be quiescent for positional comments (else defer to the end of the tactical sequence).
- `diff_vector(start, envisioned)` → signed Δcp + flag transitions (`2→1`), split into
  positive/negative lists sorted by |Δ| (Tables 5.1/5.2, kept in the move JSON for the charts
  tooltip / debugging).
- **Played-vs-best** (Guid's option 3): rework `future_line_compare.py` to return both diff
  vectors for merit comparison.
- Declarative rules (`commentary/rules/`, thresholds in data): seed set ported from §5.4.1 —
  pawn-structure improved/worsened (the X/Y/Z rule), strong-knight lost/gained, bishop pair
  eliminated, rook behind passed pawn, back-rank weakened, king-safety shift (tropism only when
  corroborated by the safety composite), piece activity. Endgame pack: king activity, passed-pawn
  metrics, opposition, outside passer, promoting existing Lucena/Philidor/fortress/zugzwang
  detectors from labels to rule inputs.
- Output per commented move — the **`CommentFacts`** object, the single source of truth:

```python
CommentFacts(
  verdict,            # "leads to equality" | "wins material" | ... ← from eval
  eval_cp, depth, engine,
  display_line,       # shortened PV (SAN list + per-step FENs)
  claims,             # fired-rule sentences, each with features_involved + Δ magnitudes
  better_alternative, # best-move CommentFacts when played != best
  flags,              # e.g. BISHOP_PAIR 2→1
)
```

- **No rule fired and no quality trigger → no CommentFacts → no comment.** Key-moment detection
  remains as prioritization (which moves get the LLM pass and at what tier), not as content.

### 2.4 LLM Enrichment Layer (replaces the freeform composer)

Prompt contract per move (one call, reverse order — §5):

- **Inviolable facts**: the CommentFacts. The comment MUST contain the verdict, the displayed
  line as a `[pv:...]` token, the eval rendered once as `({eval:+.2f}, Stockfish:{depth})`, and
  every claim (rephrased freely). It MUST NOT contain positional claims, squares, motifs, or
  lines absent from the input.
- **Enrichment inputs** (use-if-relevant): opening name/ECO + RAG opening metadata and master
  annotations (early and early-mid only), episode theme, **future context** (§5), result.
- **Humanization dial** `HUMANIZATION_LEVEL`: `0` deterministic template join (Guid baseline,
  debugging/A-B), `1` light polish (facts restated fluently, ≤1 flavor clause), `2` full flavor
  (lead-in, foreshadowing sentence, opening lore). Default `1–2` per Matej's taste; the point is
  that **all levels share identical facts**.
- Target shape (level 2):

```
{≤1 context/flavor sentence}. {move} {verdict} after [pv:#k] ({+0.74, Stockfish:16}).
{fired claims, fluently joined}. {≤1 foreshadow sentence grounded in future context}.
{Better was …  ← only when played ≠ best}
```

- Post-validation: token validator (extend `annotation_tokens.py`) checks every emitted
  `[move:]/[square:]/[piece:]/[pv:]` against the position; invalid tokens stripped; a claims
  checker verifies each named feature in the text maps to a fired rule (else regenerate once,
  then fall back to level 0 rendering). Keep the forbidden-phrase scrub. Delete the
  "use words instead of eval numbers" style rule and the archetype texture blocks.
- Episode narratives stay LLM-generated but consume the move-level CommentFacts + comments of
  their episode. `game_narrative` (string) is kept for now (FE header); **`game_summary` and the
  digest stage are scrapped** (§6).

---

## 3. Per-phase comment strategies

| Phase | Engine | Content source | Style |
|---|---|---|---|
| **early** (in book) | none | ECO entry (name, variation, code) + RAG opening metadata | 1 short sentence per *named* book transition, not per ply; one richer "out of book" comment at book exit (typical plans for the tabiya, RAG-sourced). Never mentions eval. |
| **mid** | full | CommentFacts (diff vectors + rules) | Guid format + flavor per `HUMANIZATION_LEVEL` |
| **end** | full | CommentFacts with endgame feature/rule pack | "name the decisive element"; technique motifs (opposition, Lucena…) allowed as claims because they come from detectors, not the LLM |

---

## 4. Reverse-order generation

Engine pass (phase 1) already completes before the LLM pass, so this is purely an iteration-order
change in `game_pipeline.run_llm_phases`:

1. Iterate **episodes last → first**; within an episode, commented moves **last → first**,
   sequential (a move's prompt depends on later outputs).
2. Maintain a rolling **future-context buffer** injected into each prompt:
   - final result + how the game ended (mate / flag / resignation eval),
   - the next 2–3 already-written comments verbatim (they are *later* in game time),
   - one-line synopses of all later episodes (their narratives are already generated, since we
     go backwards),
   - turning points: now computed heuristically from eval series (sign changes ≥ threshold) —
     this replaces the digest's `turning_points`.
   Cap the buffer (~250 tokens) the way the digest is capped today.
3. Episode narrative is generated **after** its moves (it consumes them), but **before** moving
   to the earlier episode (so it's available as future context).
4. Cost note: per-move parallelism is lost by design; with ~30–40 commented moves sequential
   latency is acceptable. Optional `REVERSE_PARALLEL=episode` mode: within an episode run moves
   in parallel using only episode-level future context (faster, slightly weaker foreshadowing).
5. `strategic_archetype` (currently from the digest, used for flavor): derive heuristically from
   motif trajectories + opening family, or drop — decision point during step 6 below.

---

## 5. Feature-progression charts (backend contract; widget itself is FE)

- New top-level `feature_series` in `GameJson`:

```json
"feature_series": {
  "plies": [1, 2, 3, ...],
  "features": {
    "EVALUATE_PAWNS":   {"white": [12, 12, ...], "black": [...]},
    "KING_TROPISM":     {"net":   [0, -4, ...]},
    ...
  }
}
```

  Aligned arrays over all mainline plies (book plies included — feature computation is
  engine-free). Values in centipawns. One entry per feature in §7.
- Per-move `feature_refs` on `GameMove`: the features the selected comment is grounded in —
  straight from `CommentFacts.claims[*].features_involved`:

```json
"feature_refs": [
  {"name": "EVALUATE_PAWNS", "delta_cp": -15, "direction": "black_improves"},
  {"name": "BLACK_BISHOP_PAIR", "delta_cp": 20, "flag": "2->1"}
]
```

- Per-key-moment diff tables (positive/negative vectors, Tables 5.1/5.2 style) stored under the
  move for the chart tooltip / inspection.
- FE widget (later): grid of small multiples from `feature_series`; selecting a comment
  highlights charts named in `feature_refs` and marks the ply range of `display_line`.

---

## 6. Output JSON changes (`GameJson.py`)

| Change | Detail |
|---|---|
| **add** `GameMove.phase` | `"early" / "mid" / "end"` |
| **add** `GameMove.feature_refs` | §5 |
| **add** `GameMove.feature_diff` | positive/negative diff vectors + flags for key moments |
| **add** `GameJson.feature_series` | §5 |
| **add** `GameMove.resolved_tokens` | already built at `game_pipeline.py:251`, currently dropped — emit it |
| **change** `Variation` | add per-step `fens` (and `from`/`to`) so the PV popup needs no FE-side replay; add `evals` per step where available |
| **remove** `GameJson.game_summary` | and the digest LLM stage + `GAME_SUMMARY` callback; archetype/turning-points replaced per §4 |
| keep | `game_narrative`, `episodes`, motif fields, `rag_refs` |

---

## 7. Positional feature inventory (Knowledge Module)

Per-side, centipawn-weighted, with count flags. Status against `positional_features.py`:

| # | Feature | Source | Status |
|---|---|---|---|
| 1 | `KING_TROPISM` (net) + `WHITE/BLACK_TROPISM` | Table 5.1, §5.4.2 | missing |
| 2 | `EVALUATE_KING_SAFETY` + `WHITE/BLACK_SAFETY` | §5.4.1–2 | partial (`kingExposure`, `kingShieldPawns`, unweighted, uncombined) |
| 3 | `KNIGHTS_OUTPOSTS` (+flag) | Table 5.1 | motif-only → make numeric |
| 4 | `KNIGHTS_CENTRALIZATION` (+flag) | Tables 5.1/5.2 | partial (generic `centralization`) |
| 5 | `EVALUATE_PAWNS` (aggregate) | §5.4.1 | missing |
| 6 | `WEAK_PAWNS` | Table 5.1 | partial (`backwardPawns` list) |
| 7 | `DOUBLED/ISOLATED/PASSED_PAWNS` | §5.4.1 rule | lists exist → cp-weight them |
| 8 | `PAWN_DUO` | §5.4.1 rule | missing |
| 9 | `PAWN_ADVANCES` | Tables 5.1/5.2 | missing |
| 10 | `BACK_RANK` | Tables 5.1/5.2 | missing |
| 11 | `BISHOP_PAIR` (+flag 2→1) | Fig. 5.2 | boolean → cp value + flag |
| 12 | `BISHOP_PLUS_PAWNS_ON_COLOR` | Table 5.2, Ch. 6 | missing |
| 13 | `BISHOPS_POSITION` (PSQT) | Table 5.2 | missing |
| 14 | `BISHOPS_MOBILITY` (per bishop) | Table 5.2 | partial (side-total `mobility`) |
| 15 | `ROOKS_POSITION` | Tables 5.1/5.2 | missing |
| 16 | `ROOK_ON_OPEN/HALF_OPEN_FILE` | Table 5.2 | counts → cp-weight, per rook |
| 17 | `ROOK_BEHIND_PASSED_PAWN` | §5.3.2 | missing |
| 18 | `PIECE_ACTIVITY` (W/B aggregates; Guid's additions) | Tables 5.1/5.2 | missing |
| 19 | `BAD_BISHOP` (composite; Watson weighting d/e > c/f) | Ch. 6 | heuristic motif → static feature |
| 20 | fold in existing: `centerControl`, `space`, `connectedRooks`, `weakSquares/holes`, material diff | — | add weights + sides |
| 21 | endgame pack: king activity, rule-of-the-square, outside passer, majority mobilization, opposition | Ch. 5 endgame refs | missing |

---

## 8. PGN export

- New endpoint `GET /jobs/{job_id}/pgn` rendering from the final `GameJson` (no re-analysis).
- Built with python-chess `Game`: mainline from `moves`; for each commented move:
  - **NAG** from `move_quality` / `classification` (`$1` !, `$2` ?, `$3` !!, `$4` ??, `$5` !?, `$6` ?!),
  - comment `{...}` containing `[%eval #.##]` (standard tag, Lichess/ChessBase-compatible) +
    the **plain-text rendering** of the comment (token grammar → text: `[move:Nf3]`→`Nf3`,
    `[pv:#k]` → omitted from text, becomes the RAV),
  - the displayed line as a **RAV** (variation) after the move, with a closing eval comment
    `({+0.74, Stockfish:16})`,
  - `better_alternative` as a second RAV with its own NAG/comment,
  - optional (flag `PGN_INCLUDE_FEATURES`): trailing `{Δ: EVALUATE_PAWNS −15, BISHOP_PAIR 2→1}`
    and `[%csl]`/`[%cal]` square/arrow tags derived from `square`/`piece`/`move` tokens.
- Single token-rendering module with two targets (FE JSON keeps tokens; PGN flattens them) so
  the grammar lives in exactly one place (`annotation_tokens.py`).

## 9. Interactive elements / PV popup (backend side)

- Keep grammar `pv, move, square, file, eval, piece` (`annotation_tokens.py:23`).
- **PV-by-reference**: comments emit `[pv:#k]` where `k` indexes `GameMove.variations`; the LLM
  is given the available line ids and their SAN so it never re-types lines (today it copies SAN
  into `[pv:...]`, which can drift from the actual PV). Legacy inline `[pv:san...]` remains
  parseable for old data.
- `variations[k]` carries everything one popup needs: `move_san`, full `line`, per-step
  `fens`/`from`/`to` (extend the existing `_resolve_pv_line` output), per-step or at least
  root+leaf `evals`. Slider / autoplay / step controls are pure FE on top of this.
- Emit `resolved_tokens` per move (see §6) so square/piece/move highlights need no FE parsing.

---

## 10. Implementation order

| Step | What | Main files | Depends |
|---|---|---|---|
| 0 | Cleanup: dead `trace()`, scrap digest stage + `game_summary` (+ heuristic turning points from eval series) | `engine_connector.py`, `game_pipeline.py`, `GameJson.py` | — |
| 1 | `PhaseClassifier`, engine gating for book plies, book-exit seed eval, `phase` in JSON | `analysis_retriever.py`, `key_moment_detector.py`, `GameJson.py` | — |
| 2 | `EarlyGameCommenter` (ECO + RAG opening metadata, transition comments) | new `commentary/phases/early.py`, `game_pipeline.py` | 1 |
| 3 | `guid_features.py` weighted vector + **all-ply `feature_series`** in JSON | new under `features/`, `GameJson.py` | — |
| 4 | Envisioned-position machinery: PV shortening + quiescence, diff vectors, played-vs-best; retire `pv_horizon_diff` | `analysis_retriever.py`, `future_line_compare.py` | 3 |
| 5 | Rule engine → `CommentFacts` (+`features_involved`), endgame pack | new `commentary/rules/` | 4 |
| 6 | LLM enrichment layer: fact-contract prompts, `HUMANIZATION_LEVEL`, claims/token post-validation, prompt cleanup (drop "no numbers" rule, archetype textures) | `advanced_comment_service.py`, `move_pipeline.py`, `annotation_tokens.py` | 5 |
| 7 | **Reverse-order pipeline** + future-context buffer; episode narratives folded into the backward sweep | `game_pipeline.py` | 6 |
| 8 | Output contracts: `feature_refs`, `feature_diff`, `resolved_tokens`, enriched `variations` (fens/evals), `[pv:#k]` by reference | `GameJson.py`, `game_pipeline.py` | 5, 6 |
| 9 | PGN export endpoint + token flattener | new `core/io/pgn_writer.py`, `routes/jobs.py` | 8 |
| 10 | Regenerate sample games; side-by-side old vs new; tune weights/thresholds with Matej | — | all |
| FE (later) | charts grid widget (`feature_series` + `feature_refs`), PV popup (slider/autoplay), phase styling | `aca-fe` | 8, 9 |

Steps 0–2 and 3 are independent and can start in parallel. Every step leaves the pipeline
runnable end-to-end.

## 11. Risks / open points

- **Endgame definition**: current `< 7` minor/major pieces vs. queens-off or total-material
  conventions — configurable; confirm with Matej.
- **Coverage drops by design** (no fired rule + no quality trigger → silence). Guid's stated
  trade-off; the humanization dial level 2 + early-phase opening comments keep the game feeling
  annotated. Keep the legacy freeform path behind a flag for A/B until Matej signs off.
- **Reverse sequentiality** raises wall-clock LLM time; mitigations in §4.4.
- **Claims-checker strictness**: too strict → constant level-0 fallbacks; start lenient
  (feature names + squares only), tighten with data.
- **Stockfish classical eval is gone in SF16+** — `"Use NNUE": false` may be ignored; all
  feature values come from our Python Knowledge Module, never from the engine. (The dead
  `trace()` confirmed nothing depends on engine eval terms today.)
- `game_narrative`: kept for now; cheap to drop later if Matej wants the summary family gone
  entirely.
