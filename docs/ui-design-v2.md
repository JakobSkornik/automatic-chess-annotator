# UI Design v2 — Merging feedback.txt (Matej #1) and feedback2.txt

Inputs: `feedback.txt` (UI priority: board+commentary, tabs, reasons visible,
numbered notation, 4.png line explorer), `feedback2.txt` (variation windows:
numbered notation, eval+depth at line end, key factors vs base position,
analysis board beside the base diagram; PGN export; academic debug mode),
Jakob's job-page layout request, and the current layouts in 1.png / screenshot.

Already satisfied (no work): feedback2 §2 — PGN export with comments exists
(`GET /jobs/{id}/pgn?language=...` + Export PGN button).

---

## 1. The central new concept: the Variation Inspector

Both feedback docs converge on one thing: a variation is not a row of bare SAN
chips — it is a claim about the future that must carry its **numbered line, its
evaluation and depth, its key factors, and a visual start ←→ end comparison**
(Fig. 5.2's two diagrams). One component serves every variation in the app:

```
┌─ Variation ────────────────────────────────────────────────────────────┐
│  17.Be3 Bd6 18.Nb5 Be5 19.f4 c6 20.fxe5 cxb5   (+0.27, depth 16)   ✕  │
│                                                                        │
│  ┌── Base position ──┐        ┌── After 20...cxb5 ───┐                 │
│  │   (static board)  │  ───►  │ (interactive board)  │                 │
│  │   before 17.Be3   │        │  ◀ ▶ ⏮ ⏭ ▶Play  [══●════] 6/8       │ │
│  └───────────────────┘        └──────────────────────┘                 │
│                                                                        │
│  Key factors of the final position (vs base):                          │
│  • Black's king is under pressure          KING_SHIELD −16, TROPISM +9 │
│  • White's pieces are actively placed      PIECE_ACTIVITY +18          │
│  ▸ full feature diff (before ←→ after table)                           │
└────────────────────────────────────────────────────────────────────────┘
```

- **Header**: the full line in numbered notation (`17.Be3 Bd6 18.Nb5 …`;
  Black-start lines as `17...Bd6 18.Nb5`), then `({eval:+.2f}, depth {d})` —
  feedback2's "na koncu variante se doda evaluacija in globina".
- **Two diagrams**: left static = position the line starts from; right
  interactive = steps through the line (slider, autoplay, step buttons — the
  existing PvPopup controls). The right board starts at the leaf so the default
  view IS the start↔end comparison; scrubbing explores the middle.
- **Key factors**: the claims fired on the root→leaf feature diff, each with its
  feature chips and Δcp; expandable to the full positive/negative diff table
  ("primerjava začetne in končne pozicije" + Matej's "prej ←→ potem").
- Invoked by clicking any variation anywhere: engine PV1/PV2, a comment's
  `[pv:]` token, the better-alternative line, or a card in the Lines tab.
  Rendered as a centered modal (wide enough for two boards).
- Replaces `PvPopup` (its controls move inside); `PvLineChips` keeps hover
  preview and gains numbered labels.

## 2. Game page layout (Matej's tab proposal, reconciled with the current page)

```
┌──────────────────────────────┬──────────────────────────────────────────┐
│ BOARD  (≈56vh, eval bar      │ COMMENTARY  (full-height right column)   │
│  alongside, player bars)     │ [Beginner|Intermediate|Expert]  [Debug ⌬]│
│                              │ ┌──────────────────────────────────────┐ │
│ ┌ Tabs ─────────────────────┐│ │ Move 17. Be3 — leads to equality     │ │
│ │ Moves │ Game info │       ││ │ (+0.12, Stockfish:16)                │ │
│ │ Summary │ Features │ Lines││ │ prose at selected language level     │ │
│ └───────────────────────────┘│ │ ▸ Line: 17.Be3 Bd6 18.Nb5 … [inspect]│ │
│ (tab content fills the rest  │ │ ▸ Key factors: claim rows w/ feature │ │
│  of the left column height)  │ │   chips → highlight Features tab     │ │
│                              │ │ ▸ Better was 17.Nf3 — … [inspect]    │ │
│                              │ │ ▸ Reasoning (visible in debug mode)  │ │
│                              │ └──────────────────────────────────────┘ │
│                              │ [comment index rail (prev/next, list)]   │
└──────────────────────────────┴──────────────────────────────────────────┘
```

- **Tabs under the board** (Matej's list, adapted): *Moves* (current MoveList),
  *Game info* (players/opening/engine metadata — the current two cards merged),
  *Summary* (game narrative + episode narratives), *Features* (the charts grid —
  too wide for a sidebar, correct as a tab; chart highlighting still driven by
  the selected comment), *Lines* (4.png-style grid: one card per available line
  of the current move — PV1/PV2/PV3, comment display line, better alternative —
  each card = mini board + numbered strip + "inspect" → Variation Inspector).
  Matej's proposed "Comments" tab is covered by the index rail staying in the
  right column (reading pane and index together — his "desno naj bodo
  komentarji" is the stronger constraint).
- **Right column**: the structured comment renderer fed by `comment_facts`:
  assessment header, prose at the selected language level, the displayed line
  (numbered, inspectable), key-factor rows (claims + feature chips with a
  before←→after popover), better alternative, and the Reasoning block.
- Numbered notation everywhere: chips, inspector, Lines tab, comment titles.

## 3. Debug mode (feedback2 §3 — "akademski princip")

A global toggle (commentary header, persisted in localStorage) that reveals the
derivation chain per comment — *how the system reached each conclusion*:

```
▸ Reasoning (debug)
  1. Engine: eval before +0.05 → after +0.12 (d16); best move Be3 (played).
  2. Key moment: opening_transition (priority 6); quality: best.
  3. Envisioned line: engine PV 14 plies → capped 11 → trimmed 2 (forcing tail);
     leaf quiescent ✓; start quiescent ✓.
  4. Feature diff (start → leaf): 6 positive / 4 negative entries [table].
  5. Fired rules: king_under_pressure (Δ−21 ≥ 18 ✓, tropism +9 ≥ 8 ✓),
     activity_improved (Δ+18 ≥ 14 ✓); 2 claims muted by 6-ply dedup window.
  6. Rendering: expert=llm ✓ intermediate=llm ✓ beginner=template (contract
     violation: PV token altered); forbidden-phrase hits: 0.
```

Backend: persist a `debug` block per commented move (`GameMove.debug`):
engine numbers, key-moment type + priority, envisioned-line construction stats
(raw/capped/trimmed lengths, quiescence flags), fired rules with the thresholds
they passed *and the claims muted by dedup*, and per-level rendering outcomes.
Rule engine gains a trace mode (each rule reports pass/fail with its numbers) —
this is the academically interesting artifact and costs nothing (pure python).
Gate the block behind `DEBUG_TRACE=1`... default ON: size is small and Taja's
review benefits.

## 4. Job page layout (Jakob's request, 1.png)

```
┌  left column (2/3)              ┐  ┌ right column (1/3)  ┐
│ ┌─ Engine analyzing ──────────┐ │  │ ┌─ Job details ────┐ │
│ │ stepper + progress bars     │ │  │ │ id/opening/moves │ │
│ └─────────────────────────────┘ │  │ │ provider/model…  │ │
│ ┌─ While you wait ────────────┐ │  │ │ PGN preview      │ │
│ │ (same width as card above)  │ │  │ │ (flex-1, grows)  │ │
│ └─────────────────────────────┘ │  │ └──────────────────┘ │
└─ items-stretch: equal heights ──┘  └──────────────────────┘
```

"While you wait" moves under "Engine analyzing" at matching width; "Job details"
becomes the only right-column card and stretches to the full row height, its PGN
preview box growing (`flex-1 overflow-auto`) to absorb the extra space.

---

## 5. Implementation plan

| # | Item | Repo | Notes |
|---|---|---|---|
| 1 | **Job page layout** (§4) | aca-fe | small, self-contained — do first |
| 2 | **Numbered-notation helper** `formatNumberedLine(startFen, sans)` + apply to `PvLineChips` labels | aca-fe | feeds everything below |
| 3 | **Backend: variation key factors** — at assemble time run the (engine-free) root→leaf feature diff + rules for each engine `Variation`; add `Variation.key_factors: [{text, features, delta_cp}]`, `Variation.depth`; comment lines already carry claims via `comment_facts` | backend | no new engine cost |
| 4 | **Variation Inspector** (§1) — dual board, slider/autoplay, numbered header with eval+depth, key factors, full-diff expander; wire into PvLineChips click, EvaluationPanel, alternative lines | aca-fe | replaces PvPopup |
| 5 | **Backend: debug block** — rule-engine trace (pass/fail + thresholds + muted claims), envisioned-line stats, per-level rendering outcomes → `GameMove.debug` | backend | pure refactor of existing data |
| 6 | **Game page restructure** (§2) — tabs under board, big board, right commentary column with structured renderer (assessment / line / key factors / alternative / reasoning) + debug toggle | aca-fe | largest piece; consumes 2–5 |
| 7 | **Lines tab** (4.png grid) | aca-fe | cards reuse Inspector internals |
| 8 | Jobs zip export for Taja (carried from plan v1) | backend+fe | unchanged |

Sequencing: 1 → (2,3,5 in parallel) → 4 → 6 → 7 → 8. Each step leaves the app
shippable. PGN export (feedback2 §2) is already live — nothing to do.
