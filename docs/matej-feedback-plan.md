# Plan v3 — Matej's Review (2026-06-11) vs the guid-refactor Branch

Source: `feedback.txt` + screenshots 1–4 in the workspace root. Matej reviewed the
**pre-guid-refactor** version; section 1 audits which of his points the current
`guid-refactor` branches (backend `540e9a9`, frontend `ce8d79e`) already resolve.
New requirement from Jakob: PV notation must always carry move numbers
(`20. a4 f5 21. Qb3 Bc3 ...`), and humanization must surface in the UI as a
**commentary language slider — Beginner / Intermediate / Expert** (expert = Matej's
dry dissertation style).

---

## 1. Alignment audit — what his feedback already maps to

| Matej's point | Status on `guid-refactor` |
|---|---|
| 3.png: ungrounded phrases ("where it eyes d4", "works with the queen on g3", "timing is a touch loose", "kick the h4-knight") | **Fixed.** Comments are built from `CommentFacts`; every positional claim comes from a fired rule over the feature-diff vector. The LLM contract forbids new squares/motifs/judgments; eval+PV tokens must survive verbatim or the deterministic template is used. |
| "Better silent than wrong"; short and precise | **Fixed.** No fired rule + no quality trigger → no comment; claims capped at 4 and deduped over a 6-ply window; thresholds hand-tunable (`rules/engine.py THRESHOLDS`). |
| Comment structure per Fig. 5.2 / 2.png: move + verdict + PV + (eval, engine:depth) + feature-change sentences; "A better move was [MOVE], leading to..." | **Mostly fixed.** Template emits exactly `{move} {verdict} after {PV} ({eval}, Stockfish:{depth}). {claims}. Better was {alt}...`. Missing: phrasing *variation* so the pattern doesn't repeat verbatim (his "ALI (da ni vedno ponavljajoč vzorec)") → §3.2. |
| "LLM naj predvsem pomaga ubesediti številke, ki izhajajo iz razlik med značilkami" | **Fixed by design.** That is the level-1 contract (restate fired claims fluently, add nothing). |
| Subproblems 1–4, 6 (p. 78): when to annotate, which alternatives, variation length, which features, forming annotations | **Implemented.** Key-moment/rule gating; better-alternative at ≥50 cp gap (Guid option 3); envisioned line capped at 11 plies + quiescence-trimmed; claim thresholds; template/composer. |
| Subproblem 5: feature *changes* vs *envisioned position* description | **Not implemented** — claims only use change-form ("has improved..."). → §3.3 |
| Subproblem 7: refining annotations | Partially (humanization levels); becomes the language slider → §3.1. |
| "Komentarji naj temeljijo na atributih... če bodo prikazane, bomo razumeli kateri deli izhajajo iz značilk" | **Half fixed.** `feature_refs`/`feature_diff` are in the JSON and the charts grid highlights them — but the comment itself still renders as one prose blob; the per-part "reasons" view is missing → §2.2. |
| UI: bigger board, tabs, commentary right with space | **Not done** → §2.1. |
| Move numbers in notation incl. PV | **Not done** (chips show bare SAN) → §2.3. |
| 4.png: multi-board subvariation explorer | **Not done** → §2.4. |
| JSON repository for Taja | **Not done** → §4. |
| "Game Summary" tab | **Tension:** we scrapped `game_summary` (digest) on Jakob's instruction. The tab will show `game_narrative` + episode narratives instead. Confirm with Matej that this satisfies the tab's purpose. |

Bottom line: the *generation* half of his feedback is essentially what guid-refactor
built (he reviewed the old version); the *UI* half — his stated priority — is the
bulk of the remaining work.

---

## 2. UI plan (priority, per 1.png critique)

### 2.1 Layout restructure (`aca-fe`)

Target layout (replaces the current 2×2-ish grid in `pages/game/[id].tsx`):

```
┌──────────────────────────────┬──────────────────────────────────────┐
│  BOARD (large, ~55vh,        │  COMMENTARY (full right column)      │
│  eval bar alongside)         │  ┌ language slider: Beg|Int|Expert ┐ │
│                              │  │ Move 17. Be3  (verdict, eval)   │ │
│  ┌ Tabs ───────────────────┐ │  │ ── prose (selected level) ──    │ │
│  │ Moves │ Game Info │      │ │  │ ▸ PV: 17.Be3 Bd6 18.Nb5 ...    │ │
│  │ Summary │ Comments │     │ │  │ ▸ Reasons (claims+features,    │ │
│  │ Features │ Lines │       │ │  │    before ←→ after)            │ │
│  └─────────────────────────┘ │  │ ▸ Better was 17.Nf3 (+PV)      │ │
│                              │  └ comment index / prev-next ──────┘ │
└──────────────────────────────┴──────────────────────────────────────┘
```

- **Board card grows** (≈ `min(55vh, 640px)`, eval bar beside it; Opening metadata
  collapses into "Game Info" tab).
- **Tab strip under the board** (new `Tabs` ui component):
  - *Moves* — existing `MoveList`.
  - *Game Info* — `OpeningMetadataCard` + players/result/engine info.
  - *Summary* — `game_narrative` + episode narratives (replaces scrapped digest).
  - *Comments* — index of commented moves (the right-rail list that today lives
    inside `Comments.tsx`), click-to-jump.
  - *Features* — the `FeatureChartsPanel` grid (kept; audience is chess players,
    so keep names chess-literal).
  - *Lines* — the §2.4 explorer.
- **Right column = commentary only**, full height (`Comments.tsx` rework, §2.2).

### 2.2 Structured comment renderer ("reasons" + per-part PVs)

Matej: comments may be split into parts, each part with its own PV; the reasons
must be visible (features, before ←→ after).

- Backend: emit the (trimmed) facts per commented move — new `GameMove.comment_facts`:
  ```json
  {
    "verdict": "leads to equality", "eval_cp": 12, "eval_mate": null, "depth": 16,
    "display_line": {"san": [...], "fens": [...], "start_fen": "..."},
    "claims": [{"text": "...", "text_state": "...", "features": [...], "delta_cp": 15, "flag_note": "2 -> 1"}],
    "better_alternative": {"san": "Nf3", "verdict": "...", "eval_cp": 5,
                            "display_line": {...}, "claims": [...]}
  }
  ```
  (`feature_diff` stays as-is for the before←→after table.)
- Frontend `CommentItem` rework — renders three blocks:
  1. **Assessment**: `Move 17. Be3 — leads to equality (+0.12, Stockfish:16)` +
     numbered PV chips (click → PvPopup).
  2. **Reasons**: one row per claim — sentence + feature chips
     (`EVALUATE_PAWNS −15`, `BISHOP_PAIR 2→1`); chips hover-link to the matching
     chart in the Features tab and to a compact before←→after popover fed by
     `feature_diff` (the "prej ←→ potem" ask).
  3. **Alternative**: `Better was 17.Nf3 — ...` + its own numbered PV chips.
  - The LLM prose (per selected language level) renders above the blocks;
    at Expert level the prose *is* the template so blocks may collapse into it.

### 2.3 Move numbers in all notation (Jakob's hard requirement)

Format: `20. a4 f5 21. Qb3 Bc3 ...`; a line starting with Black: `20... f5 21. Qb3`.

- New FE helper `formatNumberedLine(startFen, sans[]) -> {label, san}[]` (fullmove
  number + side derived from the start FEN; number rendered before white moves,
  `N...` prefix when the line starts with Black).
- Apply in: `PvLineChips` (chips show `21.Qb3` style labels), `PvPopup`
  (header `3/8 · 21.Qb3` + slider tick labels), the structured comment blocks,
  and the Lines explorer.
- Tokens stay raw SAN in the JSON (`[pv:...]` must remain parseable by
  `annotation_tokens._resolve_pv_line`); numbering is purely presentational.
  PGN export already numbers correctly via python-chess. Backend `resolved_tokens`
  already carry per-step FENs, so the start FEN is available everywhere.

### 2.4 Lines explorer (4.png)

New *Lines* tab for the selected move: a grid of mini-boards, one per line —
PV1/PV2/PV3 (`variations`, which now include `fens`), the comment's display line,
and the better-alternative line. Each card: mini `Chessboard` + numbered move strip
+ inline step/slider controls (reuse `PvPopup` internals as an embeddable
`PvBoardCard`). This is the "solution like 4.png" for reviewing subvariations
without leaving the page.

---

## 3. Backend plan

### 3.1 Commentary language slider (Beginner / Intermediate / Expert) — IMPLEMENTED

Final design (deliberately NOT a 1:1 mapping onto the old humanization levels):
the slider selects an **audience register**, and position-specific claims come
exclusively from the rule engine at *every* level. What scales is how much
general chess knowledge is explained around the claims:

| Slider | Register | Concept explanation | Length |
|---|---|---|---|
| **Expert** | Chess Informant / GM game-collection voice; dry, declarative; claims merged into compact compound sentences (as Fig. 5.2 itself does) | **none** — "do not explain the features"; no flavor, no narrative | ~25–55 words + tokens |
| **Intermediate** | club player (~1500–2000); standard terms used, never defined | one short clause on why the single most important claimed feature *generally* matters | ~50–85 words |
| **Beginner** | learning player; friendly instructive tone | each targeted feature explained simply on first mention (what a passed pawn *is*); jargon avoided/defined inline; one optional takeaway tied to a claimed concept | ~70–115 words |

The grounding line: *generic* chess knowledge about a concept named in a claim
("doubled pawns are hard to defend") is allowed where the audience rules permit;
*position-specific* assertions ("the c3 pawn will fall") only ever come from rules.
Enrichment (opening lore / future context) is available to intermediate+beginner;
expert ignores it.

Implementation (done):
- **One LLM call returns all three texts** (`{"expert","intermediate","beginner"}`
  JSON schema); each validated against the fact contract independently; any failing
  level falls back to the deterministic template. Expert may thus be LLM-merged
  (better flow than raw sentence-join) but degrades to the exact template.
- Template has deterministic phrasing variants by ply parity (Matej's
  anti-repetition ask) and uses envisioned-state claim forms on long quiescent lines
  (subproblem 5); rules emit both `text` and `text_state` per claim.
- JSON: `GameMove.comments`, `resolved_tokens_by_level`, trimmed
  `GameMove.comment_facts`; `comment` mirrors intermediate for compatibility.
- PGN: `?language=expert|intermediate|beginner` (default expert).
- FE: segmented control in the commentary header, persisted in localStorage;
  instant switching (all levels ship in the JSON).
- `HUMANIZATION_LEVEL=0` remains the LLM kill-switch (all levels = template).

### 3.2 Template phrasing variation (anti-repetition)

Matej explicitly wants alternation: "A better move was [MOVE], leading to
equality after..." vs "[MOVE] leads to equality". Add 2–3 deterministic phrasing
variants for the assessment head and the better-alternative head, selected by ply
hash (stable across re-renders, no RNG). Same facts, rotated surface forms.

### 3.3 Subproblem 5 — change-form vs envisioned-state-form claims

Each rule emits both forms: `text` ("Black has improved the pawn structure.") and
`text_state` ("Black's pawn structure is now improved."). Selection rule (simple,
tunable): claims tied to the *leaf* position state (flags that persist: bad bishop,
outpost, passed pawn) render state-form when the comment already contains a long
PV; change-form otherwise. Both forms ship in `comment_facts.claims` so the FE/LLM
can choose, and the composer prompt may pick whichever reads better — both are
rule-grounded so the contract is preserved.

### 3.4 PV numbering in backend text (template/PGN)

Template text keeps raw `[pv:...]` tokens (FE numbers them). The *plain-text*
template rendering used in PGN flattening already gets numbering from python-chess
RAVs — verify `1...`-style black-start lines render per the required format.

---

## 4. Review repository for Taja

Lightest useful version:
- New endpoint `GET /jobs/export.zip` — zips `data/games/*.json` (+ optional PGNs).
- Home page: "Download all (zip)" button + per-job JSON/PGN download links.
- Drive upload stays a manual step (or a `scripts/sync_drive.py` using a service
  account if wanted later). Matej only needs the files reviewable — the zip
  endpoint plus Drive is enough.

---

## 5. Order of work

| # | Item | Repo | Depends |
|---|---|---|---|
| 1 | `comments` (3 levels, single LLM call) + `comment_facts` in GameJson; phrasing variants; state-form claims | backend | — |
| 2 | `formatNumberedLine` + numbered chips/popup labels everywhere | aca-fe | — |
| 3 | Layout restructure: big board + tabs + right commentary column | aca-fe | — |
| 4 | Structured comment renderer (assessment / reasons+features / alternative) + language slider | aca-fe | 1, 2 |
| 5 | Lines explorer tab (4.png) | aca-fe | 2 |
| 6 | PGN `?language=` param; verify numbering of black-start RAVs | backend | 1 |
| 7 | Jobs zip export + download links | backend + aca-fe | — |
| 8 | Regenerate sample games at all levels; share zip with Taja; confirm "Summary tab = narrative+episodes" with Matej | — | all |

Open questions for Matej (flag, don't block):
- Summary tab content: narrative + episodes instead of the scrapped digest — OK?
- Expert voice: pure template (zero LLM) or LLM-tightened template? Plan assumes pure template.
- Does Taja review the JSONs or the rendered UI? (Affects how much to invest in §4.)
