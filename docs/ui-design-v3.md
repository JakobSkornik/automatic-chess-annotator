# UI Design v3 — Jakob's review of the v2 build (2026-06-12 screenshot)

Remarks addressed: full-height board+moves side by side, smaller board, narrower
commentary, no Debug button (feature values always visible), charts docked at a
fixed 50% of the commentary column, board-colored chart styling without a
legend, the PV player embedded instead of a popup, and — most important —
**mover-perspective comments**.

---

## 1. Mover-perspective comments (backend — the real defect)

Current failure (5.c3, a White move):

> "5.c3 leads to equality: [pv] (+0.08, Stockfish:16). **Black's bad bishop is
> no longer a problem. Black's back rank is vulnerable.** White's pieces are
> actively placed."

The rule engine fires claims for both sides indiscriminately and the composer
lists them in |Δ| order, so a White move gets explained by what it does *for
Black*. Fix at the facts level, not the prose level:

1. **`Claim.beneficiary: "white" | "black"`** — every rule already knows which
   side a claim favors (it iterates sides explicitly); record it.
2. **Ordering & framing in `build_comment_facts`**: claims favoring the
   **mover** come first — they are the move's merits. Claims favoring the
   opponent are **concessions**: cap at 1–2, tag `is_concession=True`.
   A White move's claims then read: merits ("White's pieces are actively
   placed. Black's back rank is vulnerable.") before trade-offs.
3. **Template**: merits rendered plainly; concessions prefixed deterministically
   ("In return, …" / "The cost: …" — rotated like the head variants).
4. **Composer prompt**: a MOVER PERSPECTIVE hard rule — *"Explain why the move
   serves {mover}. Claims favoring {mover} are merits; claims favoring the
   opponent are concessions — phrase them as trade-offs ('in return', 'at the
   cost of'), never as things the move accomplishes."* Applies to all three
   language levels.
5. **Variation `key_factors`** get the same beneficiary ordering relative to
   the side playing the variation's first move.
6. **Future side-switch** (Jakob will consult Matej): the `beneficiary` field
   is exactly the data a "comment on one side only" toggle needs — backend
   ready, UI deferred.

Note: claims about the opponent's *weaknesses* ("Black's back rank is
vulnerable") have `beneficiary="white"` — they are merits of the White move and
sort first. The bad-bishop claim (`beneficiary="black"`) becomes a capped,
explicitly-framed concession.

## 2. Game page layout v3

```
┌ Board ───────────────┐ ┌ Moves ───────┐ ┌ Commentary (narrower) ───────────────┐
│ board ~460px         │ │ full-height  │ │ [Beginner|Intermediate|Expert]       │
│ eval bar             │ │ move list    │ │ ┌ top: flexible, scrolls ──────────┐ │
│ ┌ Tabs ────────────┐ │ │              │ │ │ Move 5. c3 — prose               │ │
│ │ Game info │      │ │ │              │ │ │ reasons + feature values (always)│ │
│ │ Summary │ Lines  │ │ │              │ │ │ better-alternative               │ │
│ └──────────────────┘ │ │              │ │ ├ PV player (embedded, fixed h) ───┤ │
│                      │ │              │ │ │ ◀ ▶ ⏮ ⏭ ▶Play [═══●═══] 5.c3 O-O │ │
│                      │ │              │ │ ├ charts: FIXED 50% of column ─────┤ │
│                      │ │              │ │ │ feature grid (own scroll)        │ │
└──────────────────────┘ └──────────────┘ └ └──────────────────────────────────┘ ┘
```

- **Board column**: board shrinks (MAX 560 → 460); eval bar stays; tabs reduce
  to *Game info / Summary / Lines* (Moves becomes its own column; Features
  moves into the commentary column).
- **Moves column**: the move list, full height, side by side with the board.
- **Commentary column** (narrower by construction — three columns now), split
  into three fixed-purpose zones:
  1. **Comment zone** (flexible, scrollable): title + prose + the structured
     reasons with feature values **always visible** (Debug button removed).
     The verbose reasoning steps (engine numbers, classification, quiescence
     stats, muted claims) collapse into a small "details" disclosure inside the
     reasons block — visible knowledge, but not eating vertical space; this
     also fixes "section way too big for short comments".
  2. **PV player** (embedded, fixed height ≈ 300px): an always-present
     variation player — board + slider + autoplay + numbered move strip +
     `(eval, depth)` caption. Defaults to the selected move's commented line;
     clicking any variation chip (comment lines, alternative, Lines tab) LOADS
     it here instead of opening a popup. `VariationInspector` modal retired.
  3. **Charts zone** (fixed `h-[50%]` of the commentary column, own scrollbar,
     never re-flowed by comment length): the feature grid returns here,
     highlight behavior unchanged.
- Comment index rail shrinks to a slim strip (prev/next + count) above the
  comment zone; the full list stays reachable from the Moves column dots.

## 3. Chart restyle (board palette, no legend)

Per feature card, two stacked bands replace the two overlaid lines:

```
┌ Piece activity      +0.14 ┐
│ ▓▓ dark brown bg ▓▓▓▓▓▓▓ │  ← White's series: WHITE line on --board-dark
│ ░░ light brown bg ░░░░░░ │  ← Black's series: BLACK line on --board-light
└───────────────────────────┘
```

- White = white line on the dark-brown board color; Black = black line on the
  light-brown board color — self-explanatory, **legend removed**.
- Net (White-POV) features: single band, dark-brown bg, white line with the
  zero-baseline marked.
- Current-ply marker spans both bands; click-to-seek and `feature_refs`
  highlighting unchanged.

## 4. Implementation plan

| # | Item | Repo |
|---|---|---|
| 1 | `Claim.beneficiary` in rules; concession ordering/capping + `is_concession` in `build_comment_facts`; template concession phrasing; MOVER PERSPECTIVE prompt rule; beneficiary ordering for `Variation.key_factors`; serialize beneficiary in `comment_facts`/`key_factors` | backend |
| 2 | Regression tests: White-move facts put White-beneficial claims first, concessions capped + flagged; template renders "In return, …" | backend |
| 3 | `VariationPlayer` (embedded): board + controls + numbered strip + eval/depth caption; a small player-context (React state in the game page) so chips anywhere can `loadLine(...)`; chips stop opening modals | aca-fe |
| 4 | Commentary column zones: comment (flex) / player (fixed) / charts (fixed 50%, `shrink-0`); Debug button removed, reasons always on, verbose trace behind a "details" disclosure | aca-fe |
| 5 | Chart restyle: stacked white/black bands on board colors, no legend | aca-fe |
| 6 | Layout: three columns (board+tabs / moves / commentary); board MAX 460; Features tab removed, Moves tab removed | aca-fe |
| 7 | Lines tab cards load the embedded player; retire `VariationInspector` + `PvPopup` files | aca-fe |
| 8 | Build + visual pass + regenerate a sample game | both |

Order: 1–2 (backend) first — comment quality is the substance; 3–7 in one FE
pass since they touch the same column; 8 last.

## 5. Aligned suggestions (same direction, cheap now)

- **Concessions in the eval story**: when a move is a mistake/blunder, the
  opponent-beneficial claims are not "concessions" but the *explanation of the
  swing* — frame them as "This lets Black …" instead of "In return …".
  (Beneficiary + key-moment type make this a two-line rule in the template.)
- **Player follows navigation**: when the user steps through mainline moves,
  the embedded player auto-loads that move's commented line (or PV1 when no
  comment) so it is never empty — supports the "always embedded" intent.
- **Charts hover sync**: hovering a feature chip in the reasons block flashes
  the matching chart in the fixed charts zone (both are now always visible —
  the before←→after link Matej asked for becomes one glance).
- **Book moves**: the player shows the opening line (mainline continuation of
  theory) instead of nothing; charts already cover book plies.
