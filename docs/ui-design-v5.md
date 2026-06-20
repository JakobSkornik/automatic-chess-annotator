# UI Design v5 — Adopt the "chess-annotator" template (Split layout + visual system)

Source template: `automatic chess annotator/chess-annotator/` (React-18-via-CDN
prototype: `index.html`, `app.jsx`, `panels.jsx`, `board.jsx`, `styles.css`,
`data.js`, `tweaks-panel.jsx`). Target: our real app `aca-fe` (Next.js 15 +
React 19 + Tailwind), wired to the real `GameJson` backend.

Directive: take the template's **Split** layout and its **visual system**; drop
the layout switcher (Focus/Classic) and the dev tweaks panel. For the other
views (home, job) make only superficial changes so the palette/accent/fonts
match.

This supersedes the v4 chart/layout decisions where they conflict; it keeps all
the v4 *data* work (comment_facts, multi-level comments, beneficiary/concession,
refutation, transition verdicts, debug traces, numbered notation, pre-analysis
options, PGN export).

---

## 1. What the template provides (and what we take)

**Visual system (`styles.css`)** — calm, technical analysis-tool aesthetic, its
own palette (explicitly *not* KYBM):
- Token ladder: `--bg / --bg-2 / --bg-3 / --bg-inset` (backdrop → panel → inset →
  deep), `--line / --line-2` (borders), `--fg / --fg-2 / --fg-3` (text),
  `--accent (+ -2, -bg)` green, semantic `--good/--inacc/--mistake/--blunder/--info`,
  board `--board-light/dark + --board-coord + --hl-last`, `--eval-white/black`,
  `--shadow/--shadow-sm`. Full **dark + light** themes via `[data-theme]`.
- Fonts: **Hanken Grotesk** (sans) + **JetBrains Mono** (mono); `.eyebrow`,
  `.mono` utilities.
- Component chrome: `.panel` (16px radius card) + `.panel-head`; `.btn`,
  `.btn-primary`, `.icon-btn`; `.seg` segmented control; `.menu` dropdown;
  `.chip`; `.move-row/.move-cell` with quality glyph colors; `.feat-card` +
  sparkline; `.pv-card` + numbered `.pv-move` tokens; `.disclosure`; `.plate`
  player plates; vertical `.evalbar`; `.overlay/.modal`; `.tabs`.

**Split layout (`app.jsx`)** — the one we adopt:
```
grid-template-areas: "board comment moves"
                     "feat   feat   feat";
grid-template-columns: minmax(360,440) minmax(380,1.1fr) minmax(280,360);
```
Board (left) · Commentary (center, widest) · Moves (right); full-width
Positional-features strip below. Header topbar + game-info bar above the grid.

**We drop:** the `Focus`/`Classic` layouts and the layout `seg` toggle;
`tweaks-panel.jsx` entirely (a prototyping scaffold — board-style/accent live
editor + edit-mode host protocol). We may keep a *fixed* accent (template green)
and board style (Sage); an optional settings menu is out of scope for now.

**We do NOT take from the template** (our real versions are better/correct):
- `toPGN`/`toJSON` (naive) — keep our `/jobs/{id}/pgn` endpoint + JSON export.
- Fabricated `data.js` series — we have real `feature_series`, `comment_facts`,
  multi-level `comments`, `variations`, `debug`.
- The `chess.js`-in-browser board replay — we already store per-ply FENs.

---

## 2. Reconciliation with the current v4 app

| Template piece | Maps to our existing | Action |
|---|---|---|
| `.panel` + `panel-head` | `ui/Card.tsx` | restyle Card to the panel look (16px radius, header row) |
| Split grid | `pages/game/[id].tsx` | relayout to board/comment/moves + feat strip |
| `GameBar` | v4 header info strip | restyle to `.gamebar` (players · result · opening · engine · TC) |
| `BoardPanel` plates + vertical eval bar | `MainlineChessboard.tsx` + `game/EvalBar.tsx` | plates above/below; eval bar becomes **vertical, beside the board** |
| `MoveList` (2-col, eval-after, quality glyph) | `MoveList.tsx` | restyle; add eval-after column + quality glyph colors |
| `Commentary` head eval chip + title + prose | `Comments.tsx` + `CommentItem.tsx` | restyle to `.comment-*`, add eval `.chip` in head |
| `.pv-card` (mini board + numbered tokens + nav) | `VariationPlayer.tsx` (pinned player) | restyle player to `.pv-card`; this IS the embedded mini-board |
| `.disclosure` "Reasoning details" | `StructuredComment.tsx` debug `<details>` | restyle to `.disclosure` |
| `FeaturePanel` sparkline cards + "All N" modal | `FeatureChartsPanel.tsx` | **replace brown bands with the template's sparkline cards** (see §5) |
| theme toggle (dark/light) | none (we use `prefers-color-scheme`) | add explicit `data-theme` toggle, persisted |

Our richer content **survives** inside the new chrome: part cards (Main line /
Better was), concession badges, refutation sentence, transition verdicts,
per-level prose (Beginner/Intermediate/Expert), `feature_refs` highlight, debug
trace, prev/next comment nav, side-filter badge.

---

## 3. Design-token migration (the "accent changer" work)

Strategy: **add the template's token layer to `globals.css`, then re-point our
existing Tailwind tokens at it.** Existing Tailwind classes (`bg-background-*`,
`text-*`, `border-*`, `accent-*`) keep working and instantly inherit the new
palette; ported template component CSS uses the raw `--bg/--fg/...` names. One
re-skin updates all three views at once.

Add verbatim from the template (both `[data-theme="dark"]` default and
`[data-theme="light"]`): `--bg, --bg-2, --bg-3, --bg-inset, --line, --line-2,
--fg, --fg-2, --fg-3, --accent, --accent-2, --accent-bg, --good, --inacc,
--mistake, --blunder, --info, --arrow, --board-light, --board-dark,
--board-coord, --hl-last, --eval-white, --eval-black, --shadow, --shadow-sm`.

Re-point our tokens (in `globals.css`, keeping `tailwind.config.js` unchanged):

| Our token | ← template value |
|---|---|
| `--color-background-primary` (panels) | `var(--bg-2)` |
| `--color-background-secondary` (inset/hover) | `var(--bg-3)` |
| body backdrop (new) | `var(--bg)` |
| `--color-text-primary/secondary/tertiary` | `var(--fg)/(--fg-2)/(--fg-3)` |
| `--color-border-tertiary/secondary/primary` | `var(--line)/(--line-2)/(--line-2)` |
| `--accent-progress` (amber, used as marker/highlight) | `var(--inacc)` |
| `--accent-engine` (blue) | `var(--info)` |
| `--accent-commentary` (purple) | `var(--accent)` (green) — unify the "go" accent |
| `--board-light/dark` | template Sage `#E8E6D9 / #6E8C6A` (dark), light variants |
| `--color-{background,text,border}-{info,warning,success,danger}` | keep, retuned to sit on the new bg ladder |

Fonts: import Hanken Grotesk + JetBrains Mono (via `@fontsource` packages, like
the existing Montserrat imports); set `--font-sans: 'Hanken Grotesk'…`,
`--font-mono: 'JetBrains Mono'…`. Drop Montserrat.

Theme: default `data-theme="dark"` on `<html>`; a topbar toggle flips it and
persists to `localStorage` (replacing reliance on `prefers-color-scheme`, which
becomes the first-run default only).

Port the template's component CSS into a scoped stylesheet (`chess-annotator.css`
imported by `globals.css`) for the classes the game view will use (`.panel`,
`.gamebar`, `.plate`, `.evalbar`, `.move-*`, `.pv-card`, `.pv-move`,
`.disclosure`, `.feat-*`, `.chip`, `.seg`, `.menu`, `.overlay/.modal`). Keep
Tailwind for layout/spacing; use these classes for the distinctive chrome.

---

## 4. Game view rebuild (`pages/game/[id].tsx` + components)

### 4.1 Shell
- **Topbar** (`.topbar`): brand mark (♞ tile) + "Chess Annotator" + `Job …`
  (mono); right side: theme toggle (`.icon-btn` sun/moon), Flip (`.btn`), Export
  dropdown (`.menu` → Export PGN / Export JSON, wired to our real exports). Drop
  the layout `seg`.
- **Game bar** (`.gamebar`): `White (elo) – Black (elo)` · result badge · opening
  (accent) · `Stockfish d16` · spacer · event/time-control. From
  `state.pgnHeaders` + `gameJson.metadata` + `analysis_info`. Also surface the
  pre-analysis badges (level, comment-side) here or in the commentary head.

### 4.2 Split grid (no switcher)
`board | comment | moves` top row, `features` full-width below. Columns per the
template; responsive collapse to stacked on narrow widths.

### 4.3 Board panel (`MainlineChessboard` restyle + `EvalBar` rework)
- Keep **react-chessboard** (correctness, our arrows/coords already work); theme
  square colors from `--board-light/dark`, 12px rounded wrapper, last-move
  highlight + accent arrow, coords in `--board-coord`.
- **Player plates** (`.plate`) above and below: avatar (initial), name, `Elo`
  (mono), `to move` badge on the side to move. Replaces the current player bars.
- **Eval bar** (`EvalBar`): make it **vertical, in a flex row beside the board**
  (`.board-stage`), column-reverse white fill, sigmoid `1/(1+e^(-cp/100*0.42))`,
  numeric label top/bottom. Book moves → neutral 50% + "Book".
- Board status line + nav row (first/prev/play/next/last) under the board, using
  the existing mainline navigation.

### 4.4 Moves panel (`MoveList` restyle)
- `.panel` + head ("Moves", eyebrow "eval after"); `.move-row` = `30px 1fr 1fr`
  (number, white cell, black cell); `.move-cell` shows SAN (mono) + quality glyph
  (`!`/`?!`/`??` colored via `--good/--inacc/--blunder`) + eval-after (mono,
  `--fg-3`); `.active` cell uses `--accent-bg`. Quality glyph from
  `move_quality`/`classification`; eval-after from `score`.

### 4.5 Commentary panel (the merge — `Comments` + `StructuredComment` + `VariationPlayer`)
- Head: "Commentary" + eval `.chip adv` (`+0.74 · depth 16`) + level/side badges
  + prev/next ‹ n/N ›.
- Body: comment title (`Move 10. Nxf7`) + prose at the selected level
  (`.comment-text`). PV tokens already render as plain numbered text (v4).
- **Part cards** (Main line / Better was) keep the v4 selection model; selecting
  one drives the PV card below.
- **PV card** (`.pv-card`) = the pinned `VariationPlayer`, restyled: `140px` mini
  board + numbered clickable `.pv-move` tokens (`.on` = current) + nav row +
  `n/total`. This is the template's signature commentary element.
- **Reasoning** `.disclosure` ("Reasoning details") = the v4 debug trace, restyled.
- Reasons-with-feature-chips stay (hover → flash the matching chart).

### 4.6 Features panel (§5)

---

## 5. Features: adopt the template's sparkline cards (replaces v4 brown bands)

The v4 brown stacked bands were the previous answer to "weird charts"; the
template's sparkline cards are cleaner and now endorsed — switch to them.

Per `FeatureCard`:
- `.feat-card` (inset bg, 11px radius); top row = feature name + value(s)
  `+0.92 / -0.90` (white = `--fg`, black = `--fg-3`) OR a `Δ swing` pill in the
  top-movers view (`.feat-swing`, accent).
- `Sparkline` (100×34 viewBox): **white series = `--accent` line**, **black
  series = `--fg-3` line**, dashed zero baseline (`--line-2`), **amber ply
  marker** (`--inacc`). Fed from our real `feature_series` (cp → /100 pawns;
  white array, black array; net = white+black).
- **Top movers**: `FeaturePanel` shows the N features with the largest recent
  swing (|Δ over last ~3 plies|) + an **"All N"** button → `FeaturesModal`
  (4-col grid of every charted feature). Our `feature_refs` (features behind the
  current comment) get the highlight ring on top of the top-mover selection.
- Stable order in the modal (no reordering-on-highlight — the v4 complaint).

This drops the brown-band `MiniChart`/`Band` and the fixed-50% strip styling in
favor of the template's grid + modal.

---

## 6. Other views — superficial restyle only

Once §3 lands, home and job pages inherit the palette/fonts/buttons. Verify and
nudge:
- **Home** (`index.tsx`, `LandingNewAnalysis.tsx`, `RecentJobsSidebar.tsx`):
  topbar brand mark; `.panel` cards; `.btn`/`.btn-primary`; the language/side
  selects and provider/effort selects styled as the template's controls; recent
  jobs list rows; "Download all (zip)" link.
- **Job** (`JobView.tsx`): pipeline stepper, progress bars (use `--accent` /
  `--info`), status badges (`--inacc`/`--blunder`), Job-details card + PGN
  preview, "While you wait" card — all on the new tokens. Keep the v4 layout
  (while-you-wait under the pipeline card; details stretch).
- **TopBar** (`ui/TopBar.tsx`): brand mark + theme toggle shared with the game
  topbar.

No structural change to these views — colors, fonts, card/button chrome only.

---

## 7. Keep from our app (do not regress)

Real PGN/JSON export endpoints; WebSocket streaming + `COMMENTARY_COMPLETE`;
multi-level `comments` + pre-analysis level/side; `comment_facts` (verdict,
display line, claims w/ beneficiary + concession + features, better_alternative,
refutation); transition verdicts; numbered notation helper; per-move `debug`;
`feature_series`/`feature_refs`; book-move handling; offline JSON viewer.

---

## 8. Implementation order

| # | Step | Files |
|---|---|---|
| 1 | Token migration: add template tokens (dark+light) + re-point our tokens; import Hanken Grotesk + JetBrains Mono; `data-theme` + toggle; port component CSS to `chess-annotator.css` | `globals.css`, new `chess-annotator.css`, `tailwind.config.js` (font vars), `ui/TopBar.tsx` |
| 2 | Restyle `ui/Card` to `.panel`; shared `.btn`/`.seg`/`.menu`/`.chip` primitives | `ui/Card.tsx`, new `ui/Button.tsx`/`ui/Menu.tsx` |
| 3 | Game shell: topbar (brand, theme, flip, export menu — no layout seg) + `.gamebar` strip | `pages/game/[id].tsx` |
| 4 | Split grid relayout (board/comment/moves + features strip), responsive | `pages/game/[id].tsx` |
| 5 | Board panel: plates + **vertical eval bar** beside board, status + nav | `MainlineChessboard.tsx`, `game/EvalBar.tsx` |
| 6 | Moves panel restyle (eval-after col + quality glyphs) | `MoveList.tsx` |
| 7 | Commentary restyle: eval chip head, comment-body, **`.pv-card` player**, `.disclosure`; keep part cards/badges/nav | `Comments.tsx`, `StructuredComment.tsx`, `VariationPlayer.tsx`, `CommentItem.tsx` |
| 8 | Features: sparkline `.feat-card`s + top-movers + "All N" `FeaturesModal`; wire to `feature_series`; keep `feature_refs` highlight | rewrite `FeatureChartsPanel.tsx` |
| 9 | Superficial pass on home + job views | `index.tsx`, `LandingNewAnalysis.tsx`, `RecentJobsSidebar.tsx`, `JobView.tsx` |
| 10 | Build (`tsc` + `next build`), visual pass dark+light, fresh sample game | — |

Order: 1–2 first (everything depends on the token/chrome layer); 3–8 are the
game view in one pass; 9 is cosmetic; 10 verifies.

## 9. Decisions to confirm

- **Theme default**: dark (template default) with a light toggle — OK?
- **Accent**: fix to template green, or keep our amber/blue/purple semantic
  accents and only adopt green as the primary "go" accent? (Plan assumes the
  latter: green primary, keep amber as the chart/ply marker, blue as engine/info.)
- **Board renderer**: keep react-chessboard (recommended) vs. port the template's
  CSS board. Plan keeps react-chessboard.
- **Features**: confirm sparkline cards replace the v4 brown bands (this reverses
  the earlier board-palette-bands decision in favor of the template).

## 10. Note on working tree

`app/core/commentary/forbidden_phrases.py` and `rules/engine.py` carry an
uncommitted backend tweak from the prior turn (un-forbid "holds the balance";
add an advantage-shrink verdict branch). Unrelated to this UI work — commit or
discard separately.
