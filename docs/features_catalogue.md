# Positional / middlegame feature catalogue

Importance is ranked for **middlegame** usefulness (king safety and pawn structure dominate per chessprogramming.org [Middlegame](https://www.chessprogramming.org/Middlegame)). Complexity: **S** few lines on python-chess; **M** auxiliary attack/defense maps; **L** search-heavy specialized endgame logic.

| Priority | Feature | Tier | Complexity | Notes |
|----------|---------|------|------------|-------|
| 1 | King safety (shield, adjacent open/semi-open files, king-zone attacks, exposure composite) | 1 | S | Always material when queens on board |
| 2 | Passed / backward / isolated / doubled pawns (square lists) | 1–2 | S–M | Structural backbone |
| 3 | Material counts + imbalance tags | 1 | S | White–Black signed diff; never STM-relative |
| 4 | Mobility, center control | 1 | S | Per-side integers |
| 5 | Weak squares + holes (pawn cannot cover) | 2 | S | Full-board weak list in v2 |
| 6 | Outposts (occupied / available) | 1 | S | Uses weak-square + pawn protection |
| 7 | King pawn tropism | 2 | S | Mean king ↔ enemy pawn Chebyshev distance |
| 8 | Open / semi-open files + rook placement | 1 | S | Open files as file indices in JSON |
| 9 | Bishop pair / good–bad bishops / color complex | 1 | S | Counts + imbalance hooks |
| 10 | Contested squares (both sides attack) | 2 | M | Capped for density |
| 11 | Trapped pieces | 2 | M | Suppressed in pure endgames (noise) |
| 12 | Piece connectivity (defends friendly; chain count) | 2 | M | Approximates CPW “connectivity” |
| 13 | Space, centralization | 1 | S | Activity proxies |
| 14 | Batteries (Q+B diagonal, Q+R line) | 1 | S | |
| 15 | Pawn islands | 2 | S | File-run decomposition |
| 16 | Pawn structure center type + tension + breaks | 1 | S | |
| 17 | Tempo / development | 4 | S | Only emitted while position is still “in book” phase |
| 18 | Blockade on passer | 3 | M | Endgame-guarded |
| 19 | Opposite-colored bishops flag | 3 | S | Endgame-guarded |
| 20 | Wrong rook pawn + bishop color | 3 | S | Heuristic flag |
| 21 | Rule of the square (passers) | 3 | M | Only pawn-heavy material |
| 22 | Mop-up distance (king to corner) | 3 | S | Only large material swings |

## References

- [Evaluation](https://www.chessprogramming.org/Evaluation) — feature categories (material, PST, pawn structure, mobility, king safety, space, tempo, connectivity, trapped pieces).
- [Middlegame](https://www.chessprogramming.org/Middlegame) — king safety emphasis.
- [Opening](https://www.chessprogramming.org/Opening) / book phase handled via ECO longest-prefix match, not fixed move count.
- [Endgame](https://www.chessprogramming.org/Endgame) — special knowledge when material low.

All numeric features exposed to the LLM are **absolute** (per side or White POV for engine cp), never negamax “side to move” scalars.
