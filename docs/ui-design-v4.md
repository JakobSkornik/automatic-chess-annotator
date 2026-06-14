# UI Design v4 — Decluttered Dashboard (Jakob's review of v3, 2026-06-12)

Remarks: cluttered UI, over-stretched PV navigator, PV lines duplicated three
times, weird charts, useless Lines tab (and its backend cost), dead Summary
tab, the language switch must move to *pre-analysis*, a pre-game side filter,
charts must not reorder on highlight, chart values barely visible. Plus two
generation bugs found in the exported PGN.

Guiding principle for v4: **every line of analysis appears in exactly one
place, and every screen region has exactly one job.**

---

## 1. The new layout

Two reading regions on top, one chart strip on the bottom. No tabs. The line
player sits **inside the commentary column, directly under the part cards** —
the comment and the line it talks about are never apart.

```
┌ TopBar ──────────────────────────────────────────────────────────────────┐
│ Chess Annotator · Job 634a…   Yevchenko (2276) – Martinez (2404) · 0-1   │
│ English Opening (A10) · Stockfish d16        [Flip] [Export JSON] [PGN]  │
├───────────────────────────┬──────────────────────────────────────────────┤
│ BOARD (~480px)            │ COMMENTARY                                   │
│  eval bar                 │  ┌ Move 36…Bxg2+   ‹ 27/27 › ────────────┐  │
│                           │  │ prose (level chosen at submit)         │  │
│ MOVES (fills the rest of  │  ├ ● MAIN LINE 36…Bxg2+ 37.Kg1 (#4, d16) ┤  │
│ the column, 2-col list,   │  │   • reasons w/ feature chips           │  │
│ scrolls)                  │  ├ ○ BETTER WAS 36.Qf2 (−6.5, d16) ───────┤  │
│                           │  │   • reasons                            │  │
│                           │  ├ ▸ reasoning details (collapsed) ───────┤  │
│                           │  ├ LINE PLAYER (selected card's line) ────┤  │
│                           │  │ ┌mini board┐ 36…Bxg2+ 37.Kg1 (#4,d16) │  │
│                           │  │ │  (208px) │ [⏮ ◀ ▶Play ▶ ⏭]          │  │
│                           │  │ └──────────┘ slider (max-w 280px)      │  │
│                           │  └────────────────────────────────────────┘  │
├───────────────────────────┴──────────────────────────────────────────────┤
│ FEATURES (full width, one row, h-scroll, fixed order, big values)        │
│ ┌Pawn Passed −0.10┐ ┌Material −5.30┐ ┌King safety 0.16┐ …               │
└──────────────────────────────────────────────────────────────────────────┘
```

What changed and why:

- **Game-info tab → header strip.** Players, Elos, result, opening, engine
  live in one line under the TopBar. The whole tab bar disappears (Lines and
  Summary are removed, see §3), which deletes a region and its chrome.
- **Moves go back under the board.** With the tabs gone the left column has
  exactly two jobs: position (board) and navigation (move list).
- **Commentary = prose + one card per line + the player, stacked.** A line
  is rendered once, as a selectable **part card** — MAIN LINE and (when
  present) BETTER WAS — with its numbered line, `(eval, depth)` and reasons.
  Selecting a card loads the **player pinned at the bottom of the same
  column**, so eyes move one card-height, not across the screen. Navigation
  re-selects MAIN LINE. No chips inside the prose; no Lines tab.
- **The player is compact**, not stretched: mini board 208px, slider capped
  at ~280px, controls beside the board — total height ≈ 230px, `shrink-0`.
- **Charts get the full-width bottom strip** (one row, horizontal scroll,
  stable order) — they gain width without competing with the comment.

## 1.5 Making the comments say more (grounded, not hallucinated)

Diagnosis from the exported PGN (`game_634a2b80….pgn`):

- *Empty comments*: "4...e6 leads to equality (+0.12, Stockfish:16)." —
  verdict + eval and nothing else. ~40 % of commented moves carry zero claims.
- *Tiny repetitive vocabulary*: "no longer has the advantage of the bishop
  pair" ×8, "pieces become more passive/active" ×10, "bad bishop no longer a
  problem" on move 3 — where Black's b7 bishop was never bad (flag flicker
  along the envisioned line).
- *No move tells you WHY at the board level*: 36.Qe6?? gets "A better move
  was Qf2" — but not that the played move runs into 36…Bxg2+ with mate.
- *Alternative blocks are noise*: "A better move was Qg3, leading to a clear
  advantage for White. … White's pieces become more passive." — alt claims
  ignore concession framing (rendering bug) and repeat the main claims.
- *Absurd priorities at decisive evals*: "leads to forced mate for Black.
  Black obtains a passed pawn." — a passed pawn next to mate-in-4.

Five mechanisms, all deterministic (the LLM only phrases them — the fact
contract is unchanged):

1. **Transition verdicts.** Verdict from (eval_before → eval_after) instead
   of the absolute eval only: "throws away a clear advantage (+0.77 → −0.43,
   Stockfish:16)", "increases the pressure (−1.5 → −3.2)", "holds the
   balance". A small phrase table keyed on the before/after brackets and the
   mover; the biggest readability win per line of code.
2. **Grounded specifics in claims.** Rules currently emit side-level
   abstractions because the feature vector stores counts. Extend the rule
   context with the start/leaf *boards* so claims name squares and files —
   "Black obtains a passed pawn **on e5**", "doubled **c-**pawns", "rook
   reaches the seventh rank **(d7)**", "bad bishop **on b7**", "strong knight
   **on d5**". All extractable from the same position objects the features
   came from; zero new analysis, zero hallucination surface.
3. **Forcing-content claims.** Deterministic scan of the display line's SAN
   (checks, captures, promotions, mate score) → claims like "the line is
   forcing: 36…Bxg2+ 37.Kg1 with mate in 4" — and for mistakes/blunders a
   **refutation claim** naming the first punishing move ("punished by
   36…Bxg2+"). This is what the 36.Qe6?? comment is missing.
4. **Claim hygiene.**
   - At decisive evals (|cp| > 500 or mate): drop positional claims entirely;
     keep verdict + forcing claim. A passed pawn does not matter in mate-in-4.
   - `bad_bishop_solved` / `bishop_pair` style flicker: a "solved/lost" claim
     only fires when the start position actually had the condition with a
     meaningful value (e.g. start flag set AND start value ≤ −15), not when
     the envisioned line briefly toggles a flag.
   - Alternative-block claims: beneficiary-filtered to **mover merits only**
     (max 2, state-form: "keeping the pieces active"), and any claim already
     present in the main block is not repeated.
5. **Concession join (PGN bug 2 below) + richer LLM input.** With specifics
   (2) and forcing content (3) in the facts, the single-level composer call
   has real material to write a readable paragraph — same contract, more to
   say.

## 2. Charts redesign (clean, stable, readable)

The v3 stacked brown bands read as noise. v4: one panel per feature —

- single **light-brown** plot area (`--board-light`), **white line** for
  White's series, **black line** for Black's series, both visible on that
  background; net features: one black line + dotted zero baseline. No legend.
- **Fixed order, never resorted** — highlight is a ring + raised border only
  (the jumping order while navigating is gone).
- **Value text doubled**: current value(s) in `text-sm font-bold` (≈13px,
  was 9px), the title stays small. The Δ badge from `feature_refs` sits next
  to the value, not in a second row.
- Current-ply marker + click-to-seek + chip-hover flash unchanged.

## 3. Removals (UI and backend)

| Removed | Backend consequence |
|---|---|
| Lines tab | `LinesPanel.tsx` deleted; **`Variation.key_factors` computation removed** (3 feature-diff+rules passes per move bought nothing visible — the played/alternative lines keep their claims via `comment_facts`); `Variation.depth` stays (used by the player caption) |
| Summary tab | `SummaryPanel.tsx` deleted; **`generate_game_narrative` LLM stage removed** (cost), `GAME_NARRATIVE` WS replaced by a `COMMENTARY_COMPLETE` event; episode narratives stay *internal only* (they feed the reverse-order future context) |
| In-game language switch | see §4 — the level is a job parameter now |
| Comment index rail / tabs / GameInfo panel | header strip + ‹ n/N › nav in the comment header |

## 4. Pre-analysis options (job parameters)

The submit form gains two selects, stored on the job and applied **during
generation** (this also fixes "the switch does nothing": there is exactly one
generated text, at the requested level):

1. **Commentary language**: Beginner / Intermediate / Expert (default
   Intermediate). The composer generates *only that level* — single-field
   schema, one register block. Cheaper than the 3-level call; `comments`
   keeps `{<level>: text}` so the JSON stays self-describing, `comment`
   mirrors it. The game page shows the level as a read-only badge.
2. **Comment side** (optional): White / Black / Both (default Both). When a
   side is chosen, only that side's moves get key-moment commentary (the
   reverse sweep simply skips the other side's moves; opening comments and
   the eval data are unaffected). `comment_side` is stored in the JSON
   metadata so the UI can label it.

Plumbing: `SubmitPgnRequest` → `queue_manager.add_job` → job dict → worker →
`GameAnnotationPipeline.run_llm_phases(level=…, side=…)`.

## 5. Generation bugs found in the exported PGN

1. **Duplicate continuation RAV**: `4. Nf3 (4. Nf3 e6 …)` — when the
   commented line's continuation starts with the very move that was played
   next in the game, the RAV duplicates it. Fix in `pgn_writer`: if the
   attached line's first SAN equals the next mainline move, drop that ply and
   attach the remainder one node deeper (or skip the RAV when ≤2 plies remain).
2. **Second concession loses its framing**: "In return, White's bad bishop is
   no longer a problem. White's pieces are actively placed." — the second
   concession reads as a merit. Fix in the template: join all concessions into
   ONE sentence — "In return, White's bad bishop is no longer a problem and
   White's pieces are actively placed." (same for "Now …" consequence mode);
   give the composer prompt the same instruction.

## 6. Implementation plan

| # | Item | Repo |
|---|---|---|
| 1 | Job params `commentary_level` + `comment_side`: request model, queue, worker, pipeline; single-level composer call; side-gated reverse sweep; metadata fields | backend |
| 2 | Remove `Variation.key_factors` computation + `generate_game_narrative` stage; `COMMENTARY_COMPLETE` WS event | backend |
| 3 | **Comment informativeness (§1.5)**: transition verdicts; grounded specifics (squares/files in claims); forcing/refutation claims; claim hygiene (decisive-eval suppression, start-state gating for solved/lost claims, alt-block merits-only + dedup) | backend |
| 4 | PGN fixes: dedupe continuation RAV; single-sentence concessions (template + prompt) + tests | backend |
| 5 | Submit form: level + side selects; job page shows them | aca-fe |
| 6 | New layout: header info strip; board+moves left; commentary column = prose / part cards / **player pinned under the cards**; charts in a full-width bottom strip | aca-fe |
| 7 | Charts: single light-brown panel, white/black lines, big values, fixed order, ring-only highlight | aca-fe |
| 8 | Delete LinesPanel/SummaryPanel/Tabs usage/level slider; `COMMENTARY_COMPLETE` handling; prose tokens render as plain styled text (no chip duplication) | aca-fe |
| 9 | Builds, tests, fresh sample game, visual pass | both |

Order: 1–4 backend (one commit), then 5–8 frontend (one commit), then 9.
