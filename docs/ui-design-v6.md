# Plan v6 — Fixes from the `game_ad65c…` review

Investigated against the referenced JSON (`game_ad65c151-…json`, 55 moves, 33
commented) and the current code. Findings drive each fix; the two "needs
research" items (rule coverage, PV-feature charts) are scoped at the end.

## Data findings (what the JSON actually shows)

- **27 of the commented mid/end moves render as the heuristic stub**
  ("Piece activation (Eval swing: +0.32 pawns, White POV step)") and **19
  `comment_facts` carry zero claims.** So the dominant quality problem is
  *rules rarely fire* and the fallback is the ugly stub — not the LLM.
- **No saved comment contains the JSON-schema blob** — the schema-leak the
  user pasted comes from the **live stream / LLM-debug**, not the saved field.
  But the leak is real: `_parse_text` does a single `json.loads(raw)`; when the
  provider echoes the schema and/or concatenates objects, it throws and we
  return the *entire blob*, which then **passes** `validate_facts_comment`
  (the eval + PV tokens are present inside it) and becomes the comment.
- **`debug.passes` / `system_prompts` / `tier` / `rationale` / `rag_query` are
  absent** from the v4 facts `llm_debug` (it only has `facts_renderings`,
  `facts_contract_ok`, `forbidden_phrase_hits`, `claims`, `feature_refs`), but
  `LlmDebugPanel` calls `.map()` on them unconditionally → TypeError → the
  panel crashes/won't open.
- `feature_series` is **mainline-only**; the PV/`display_line` carries `fens`
  but **no per-ply feature values** → nothing to chart for the PV yet.
- "Fired rules: none" appears on genuine mistakes/blunders (11…Nd5, 13…Ba4,
  15.Qxe6+, 26…Nd5, 27.Bxd5) — tactical swings the positional rules don't model.

---

## Remark-by-remark fixes

### A. LLM debug panel crashes (remark 1)  — FE
Root cause: unconditional `.map()` over fields the facts pipeline never sets.
Fix: rewrite `LlmDebugPanel` to the **facts shape** and guard everything.
Render what exists: `facts_renderings` (per-level llm/template), `contract_ok`,
`forbidden_phrase_hits`, `claims[]`, `feature_refs[]`; fall back to a generic
`<pre>{JSON.stringify(debug, null, 2)}</pre>` for unknown shapes so it can
never throw. (Legacy `passes`/`system_prompts` still rendered *if present*.)

### B. Reasoning open by default (remark 2) — FE
`StructuredComment`'s `<details className="disclosure">` → add `open`
(`<details open>`). One-line change.

### C. Selects don't open on click (remark 4) — FE
Native `<select>` in `LandingNewAnalysis` only responds to arrow keys (a known
click-swallow under our layout/theme). Replace with a small template-styled
**`ui/Select` dropdown** (button + `.menu`, click-to-open, outside-click close —
same pattern as the Export menu). Use it for language, side, provider, effort.
Bonus: matches the chess-annotator look and is keyboard-accessible.

### D. Schema-leak / incorrect parsing (remark 5) — backend
Harden `composer._parse_text`:
1. Try `json.loads` (the happy path).
2. Else scan for **all** `{...}` balanced objects in `raw`; take the **last**
   one that parses to a dict with a non-empty `text` field.
3. Else strip lines that are obviously the schema (`"additionalProperties"`,
   `"type":"object"`, `"properties"`) and any leading/trailing braces, then use
   what remains.
Also tighten `validate_facts_comment`: **reject** any candidate that still
contains `additionalProperties`, `"type":"object"`, or starts with `{` — so a
blob can never pass even if extraction misfires (forces the template fallback).

### E. Layout rework of the commentary PV area (remark 3) — FE + backend

Target shape of the pinned PV card (replaces the current board-left /
tokens-right grid):

```
┌ Main line / Better was            (+5.31, depth 16)  5/10 ┐
│ ┌────────────┐   ┌ FIRED-RULE CHARTS (this comment) ────┐ │
│ │  PV board  │   │ Piece Activity   mainline ~~ | PV ~~  │ │
│ │  (board W) │   │ Bad Bishop       mainline ~~ | PV ~~  │ │
│ ├────────────┤   │ …one row per feature_ref…             │ │
│ │ ──slider── │   └───────────────────────────────────────┘ │
│ │ (board W)  │                                              │
│ ├────────────┤                                              │
│ │ ⏮ ◀ ▶ ▶ ⏭ │   numbered move tokens (wrap under)          │
│ └────────────┘                                              │
└─────────────────────────────────────────────────────────────┘
```

- **Slider below the board, board-width**: move the range input directly under
  the mini board, `width = board width`, with the nav buttons under it — the
  board+slider+controls become one tight vertical stack (left), freeing the
  right side for charts. (Interpreting "make it as tight as the board, so it's
  vertical" = the *PV card becomes a vertical stack*, slider constrained to
  board width. If you literally want a rotated vertical slider, say so.)
- **Fired-rule charts beside the board (right)**: for each feature in the
  comment's `feature_refs` / `claims[].features`, render a compact chart row.
  Each row shows **two sparklines**: the **mainline** progression (from
  `feature_series`, current ply marked) and the **PV** progression along the
  displayed line (new data, §F). Seeing both makes the fired rule legible:
  "this feature is what the line changes."
- Move-token line wraps under the charts (kept, numbered, click-to-seek the PV).

### F. Backend: features along the PV (remark 3, "calculate PV features") — backend
The guid feature vector is engine-free, so computing it at each `display_line`
FEN is cheap. Add to `build_comment_facts`:
- `EnvisionedLine.feature_series`: `{ "<FEATURE>": [cp per kept ply] }` for the
  charted features, computed from `display_line.fens` (and the start FEN).
- Same for `better_alternative.display_line`.
Serialize in `comment_facts` JSON. FE charts read it directly. (No engine cost;
~11 vector computations per commented move.)

### G. Rules don't explain tactical mistakes (remark 6) — backend, "research"
When `claims == []` (esp. mistakes/blunders/inaccuracies), the comment has no
"why". Add a **deterministic tactical/eval fallback claim** in
`build_comment_facts` when no positional rule fired:

1. **Material/eval delta**: compare material (from the feature vector) start→leaf
   and best-line leaf. If the played line drops material the best line keeps,
   emit a claim: *"drops material — {best} held the balance"* (grounded: the
   material feature delta + the best move SAN).
2. **Refutation-shaped**: if `refutation_san` exists and the line is forcing
   (checks/captures in `display_line`), emit *"runs into {refutation}, the
   tactic that punishes it"*.
3. **Otherwise eval-only**: *"the engine prefers {best} ({best_eval}); {played}
   concedes {Δ}"* — a verdict-grade sentence so the LLM/template always has
   something concrete beyond the bare verdict.

These are **facts** (eval numbers + SAN from the engine pass), not LLM
inference, so the fact contract holds. This converts the 19 zero-claim /
"Fired rules: none" comments into informative ones and removes most heuristic
stubs. Mark them as a distinct `rule_id` (`tactical_loss` / `eval_concession`)
so the debug trace shows the derivation.

### H. Floor the fallback at template facts (root of the 27 stubs) — backend
Ensure assemble/move-pipeline **never** emits the "X (Eval swing: …)" stub for
a move that has `comment_facts`: the floor must be `render_facts_template`
(verdict + numbered line + eval), which is already strictly better. Audit the
comment-resolution order in `assemble_game_json` + `FactsComposeStage` so a
facts move always carries at least the template text.

---

## Implementation order

| # | Item | Repo | Notes |
|---|---|---|---|
| 1 | `LlmDebugPanel` rewrite (guarded, facts shape) | aca-fe | fixes the crash; unblocks debugging the rest |
| 2 | `<details open>` reasoning | aca-fe | trivial |
| 3 | `ui/Select` dropdown + swap the 4 selects | aca-fe | click-to-open |
| 4 | `_parse_text` hardening + `validate_facts_comment` reject blobs | backend | + tests |
| 5 | Floor fallback at template facts (no stub for facts moves) | backend | + tests |
| 6 | Tactical/eval fallback claims when no rule fires | backend | + tests; the substance fix for remark 6 |
| 7 | PV feature_series in `EnvisionedLine` (+ alt) | backend | engine-free |
| 8 | PV card relayout: vertical board+slider+controls; fired-rule charts (mainline + PV sparklines) beside it | aca-fe | consumes §F |
| 9 | Build (tsc + next build), regen a sample, verify all remarks | both | |

Order: backend 4→7 first (comment content is the substance and the FE charts in
§8 depend on §7), interleave the quick FE fixes 1–3 early so the UI is usable
while iterating.

## Decisions to confirm
- **Slider**: PV card as a vertical stack with a board-width horizontal slider
  (assumed), or a literally rotated vertical slider?
- **Fired-rule charts**: show one row per `feature_ref` (features the comment is
  grounded in). For zero-claim comments (after §G), show the material/eval
  feature instead so the panel is never empty. OK?
- Regenerate the sample game on current HEAD before judging comment quality —
  the referenced JSON predates the v4 single-level + template-floor changes.
