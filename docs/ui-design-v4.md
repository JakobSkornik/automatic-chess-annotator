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

Two reading regions on top, one instrument strip on the bottom. No tabs.

```
┌ TopBar ──────────────────────────────────────────────────────────────────┐
│ Chess Annotator · Job 634a…   Yevchenko (2276) – Martinez (2404) · 0-1   │
│ English Opening (A10) · Stockfish d16        [Flip] [Export JSON] [PGN]  │
├───────────────────────────┬──────────────────────────────────────────────┤
│ BOARD (~480px)            │ COMMENTARY (reading pane)                    │
│  eval bar                 │  ┌ Move 36…Bxg2+   ‹ 27/27 › ───────────┐   │
│                           │  │ prose (one paragraph, generated at    │   │
│ MOVES (fills the rest of  │  │ the level chosen at submit)           │   │
│ the column, 2-col list,   │  └────────────────────────────────────────┘  │
│ scrolls)                  │  ┌ ● MAIN LINE  36…Bxg2+ 37.Kg1 (#4, d16) ┐  │
│                           │  │   • Black obtains a passed pawn        │  │
│                           │  │     PAWN_PASSED +10 [0→1]              │  │
│                           │  ├ ○ BETTER WAS 37.Qe2  (+0.4, d16) ──────┤  │
│                           │  │   • …reasons…                          │  │
│                           │  └ ▸ reasoning details (collapsed) ───────┘  │
├───────────────────────────┴──────────────────────────────────────────────┤
│ LINE PLAYER (left, compact, fixed w) │ FEATURES (rest, one row, h-scroll │
│ ┌ mini board ┐ 36…Bxg2+ 37.Kg1      │ ┌Pawn Passed┐┌Material┐┌King…┐    │
│ │  (208px)   │ (#4, depth 16)       │ │  -0.10    ││  -5.30 ││ 0.16│ …  │
│ └────────────┘ [⏮ ◀ ▶Play ▶ ⏭]     │ │ ~~~chart~~││ ~~~~   ││ ~~~ │    │
│   slider (max-w 280px)              │ └───────────┘└────────┘└─────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

What changed and why:

- **Game-info tab → header strip.** Players, Elos, result, opening, engine
  live in one line under the TopBar. The whole tab bar disappears (Lines and
  Summary are removed, see §3), which deletes a region and its chrome.
- **Moves go back under the board.** With the tabs gone the left column has
  exactly two jobs: position (board) and navigation (move list). The board
  stays ~480px; the move list takes the remaining height.
- **Commentary = prose + one card per line.** The duplication dies here: a
  line is rendered once, as a selectable **part card** — MAIN LINE and (when
  present) BETTER WAS — each with its numbered line, its `(eval, depth)` and
  its reasons underneath. No chips inside the prose paragraph (tokens render
  as plain styled text), no separate "LINE" row, no Lines tab.
- **Selecting a part card drives the player.** The bottom-left line player
  always shows the selected card's line (default: MAIN LINE; navigation
  re-selects it). One line, one place to read it, one place to play it.
- **The player is compact**, not stretched: fixed mini board (208px), the
  slider capped at ~280px, controls under the caption. It sits in the
  bottom strip, not inside the commentary column.
- **Charts form the rest of the bottom strip**: one row (two on tall screens),
  horizontally scrollable, stable order.

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
| 3 | PGN fixes: dedupe continuation RAV; single-sentence concessions (template + prompt) + tests | backend |
| 4 | Submit form: level + side selects; job page shows them | aca-fe |
| 5 | New layout: header info strip, board+moves left column, commentary part cards (selection ↔ player), bottom strip (compact player + charts row) | aca-fe |
| 6 | Charts: single light-brown panel, white/black lines, big values, fixed order, ring-only highlight | aca-fe |
| 7 | Delete LinesPanel/SummaryPanel/Tabs usage/level slider; `COMMENTARY_COMPLETE` handling; prose tokens render as plain styled text (no chip duplication) | aca-fe |
| 8 | Builds, tests, fresh sample game, visual pass | both |

Order: 1–3 backend (one commit), then 4–7 frontend (one commit), then 8.
