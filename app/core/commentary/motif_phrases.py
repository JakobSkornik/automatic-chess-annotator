"""Coach phrasing hints per motif (for prompts + rationale)."""

from __future__ import annotations

from collections.abc import Sequence

from app.models.chess_events import StrategicMotif, TacticalMotif

_OVERRIDES: dict[str, str] = {
    "fork": "double attack — one piece attacks two enemy units at once",
    "pin": "pin — a piece cannot move without exposing a more valuable target",
    "skewer": "skewer — attack through a valuable piece to a second target behind it",
    "removal_of_guard": "removal of the guard — capture or drive away a key defender",
    "deflection": "deflection — lure a defender away from a duty it was performing",
    "interference": "interference — block the line between two enemy pieces",
    "discovered_attack": "discovered attack — moving one piece unmasks an attack along a line",
    "discovered_check": "discovered check — the same, with check",
    "sacrifice": "material sacrifice for compensation or attack",
    "zwischenzug": "in-between move (zwischenzug) that changes the sequence",
    "trade_to_defuse_attack": "trade or capture to take the sting out of an attack",
    "quiet_move_threat": "quiet move that creates a strong threat",
    "color_complex_weakness": "weakness on light or dark squares around the king or camp",
    "central_counter_vs_wing_attack": "central counter against a wing attack",
    "opposite_side_castling_race": "opposite-side castling — both sides rush opposite-wing attacks",
    "wrong_wing_piece_in_race": "piece drifts to the wrong wing in a race",
    "outpost_occupied": "knight or piece occupies a strong outpost",
    "outpost_available": "outpost square is available for occupation",
    "outpost": "outpost — a square deep in enemy turf hard to challenge",
    "prophylaxis": "prophylaxis — prevents the opponent's idea before it starts",
    "luft": "luft — a pawn or king move that gives the king breathing room",
    "weak_square_created": "creates a weak square complex in the opponent's camp",
    "weak_square_exploited": "exploits an existing weak square or complex",
    "weak_square_creation": "weak squares appear in the opponent's position",
    "good_bishop": "active bishop not blocked by its own pawns",
    "bad_bishop": "bishop hemmed in by its own pawn chain",
    "bishop_pair_advantage": "two bishops vs bishop+knight or two knights",
}


def _build_glossary() -> dict[str, str]:
    out: dict[str, str] = {}
    for e in list(TacticalMotif) + list(StrategicMotif):
        key = e.value
        out[key] = _OVERRIDES.get(
            key,
            f"the pattern '{key.replace('_', ' ')}' — name it in plain chess words with this position's details",
        )
    return out


MOTIF_GLOSSARY: dict[str, str] = _build_glossary()


def glossary_phrase_for(motif_key: str) -> str:
    return MOTIF_GLOSSARY.get(
        motif_key,
        f"the idea '{motif_key}' — describe with concrete squares and pieces",
    )


def motif_glossary_prompt_block() -> str:
    lines = [
        "MOTIF GLOSSARY (use these ideas in prose; never dump this list as labels):",
        "If a motif is NOT listed under DETECTED_MOTIFS for this move, do not name it.",
    ]
    for k in sorted(MOTIF_GLOSSARY.keys()):
        lines.append(f"- {k}: {MOTIF_GLOSSARY[k]}")
    return "\n".join(lines)


def motif_glossary_prompt_block_for(keys: Sequence[str]) -> str:
    """Per-move glossary: only listed motif keys (falls back to full glossary if none)."""
    uniq = sorted({str(k).strip() for k in keys if str(k).strip()})
    if not uniq:
        return motif_glossary_prompt_block()
    lines = [
        "MOTIF GLOSSARY (subset for this move; weave naturally — never dump as a labeled list):",
        "Only name motifs that appear under DETECTED_MOTIFS.",
    ]
    for k in uniq:
        lines.append(f"- {k}: {glossary_phrase_for(k)}")
    return "\n".join(lines)
