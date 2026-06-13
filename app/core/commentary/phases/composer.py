"""Guid-format comment composition at three audience levels.

Every comment follows the dissertation's shape (Fig. 5.2):

    {move} {verdict} after {variation} ({eval}, Stockfish:{depth}).
    {fired-rule claims}. [Better was {best move}: {its claims}.]

The **commentary language level** is an audience register, not a licence to
invent. Position-specific claims come exclusively from the rule engine at
every level; what scales with the level is how much general chess knowledge
is explained around them:

  expert        Informant/Chessbase register for strong players. Dry,
                declarative, no didactics, claims merged into compact
                compound sentences. Matej's dissertation voice.
  intermediate  Club player (~1500-2000). Same facts; one brief clause on
                why the key feature matters in general. Standard terms used,
                never defined.
  beginner      Learning player. Each named feature explained in plain words
                (what a passed pawn IS), jargon avoided or defined inline,
                one short takeaway tied to a fired claim's concept.

All three texts are produced by ONE LLM call returning a JSON object with
all levels; each text is validated against the fact contract (eval + PV
tokens verbatim, alternative named) independently, and any failing level
falls back to the deterministic template. With no LLM configured (or
``HUMANIZATION_LEVEL=0``) every level is the template.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.models.comment_facts import BestAlternative, Claim, CommentFacts

logger = logging.getLogger(__name__)

LEVELS = ("expert", "intermediate", "beginner")
DEFAULT_LEVEL = "intermediate"


def humanization_level() -> int:
    """Legacy switch: 0 disables the LLM entirely (template-only output)."""
    try:
        lvl = int(os.environ.get("HUMANIZATION_LEVEL", "2"))
    except ValueError:
        lvl = 2
    return max(0, min(2, lvl))


def llm_rendering_enabled() -> bool:
    return humanization_level() > 0


# ---------------------------------------------------------------------------
# Shared fact tokens
# ---------------------------------------------------------------------------

def eval_token(facts: CommentFacts) -> str:
    if facts.eval_mate is not None:
        return f"(#{abs(facts.eval_mate)}, {facts.engine}:{facts.depth})"
    if facts.eval_cp is None:
        return ""
    # Show the eval STORY when the move moved the needle.
    if (
        facts.eval_before_cp is not None
        and abs(facts.eval_cp - facts.eval_before_cp) >= 50
    ):
        return (
            f"({facts.eval_before_cp / 100:+.2f} → {facts.eval_cp / 100:+.2f}, "
            f"{facts.engine}:{facts.depth})"
        )
    return f"({facts.eval_cp / 100:+.2f}, {facts.engine}:{facts.depth})"


def pv_token(facts: CommentFacts) -> str:
    line = facts.display_line
    if not line or not line.line_san:
        return ""
    return "[pv:" + " ".join(line.line_san) + "]"


def _alt_pv_token(alt: BestAlternative) -> str:
    if not alt.display_line or not alt.display_line.line_san:
        return ""
    return "[pv:" + " ".join(alt.display_line.line_san) + "]"


def _move_label(facts: CommentFacts) -> str:
    move_no = (facts.ply + 1) // 2
    dots = "." if facts.mover == "White" else "..."
    return f"{move_no}{dots}{facts.san}"


def _gerundize(verdict: str) -> str:
    """'leads to equality' -> 'leading to equality' (for 'A better move was X, ...')."""
    for head, ger in (("leads", "leading"), ("gives", "giving"), ("leaves", "leaving")):
        if verdict.startswith(head + " "):
            return ger + verdict[len(head):]
    return verdict


# ---------------------------------------------------------------------------
# Deterministic template (expert fallback / no-LLM rendering)
# ---------------------------------------------------------------------------

def _claim_text(c: Claim, *, prefer_state: bool) -> str:
    if prefer_state and c.text_state:
        return c.text_state
    return c.text


def render_facts_template(facts: CommentFacts) -> str:
    """Guid-format rendering with deterministic phrasing rotation (per ply),
    so the pattern does not repeat verbatim move after move."""
    variant = facts.ply % 2
    pv = pv_token(facts)
    ev = eval_token(facts)
    move = _move_label(facts)

    parts: List[str] = []
    if variant == 0:
        head = f"{move} {facts.verdict}"
        if pv:
            head += f" after {pv}"
        if ev:
            head += f" {ev}"
    else:
        head = f"{move} {facts.verdict}"
        if pv:
            head += f": {pv}"
        if ev:
            head += f" {ev}"
    parts.append(head + ".")

    if facts.refutation_san:
        parts.append(f"The move is punished by {facts.refutation_san}.")

    if facts.claims:
        # Long quiescent lines describe the envisioned position -> state form.
        prefer_state = bool(
            facts.display_line and len(facts.display_line.line_san) >= 6
        )
        merits = [c for c in facts.claims if not c.is_concession]
        concessions = [c for c in facts.claims if c.is_concession]
        if merits:
            parts.append(" ".join(_claim_text(c, prefer_state=prefer_state) for c in merits))
        if concessions:
            if facts.concession_mode == "consequence":
                # Claims start with the side's name, so "Now Black ..." reads
                # naturally — but use the change-form to avoid "Now ... is now".
                prefix = "Now " if variant == 0 else "The drawback: "
                clauses = [c.text.rstrip(".") for c in concessions]
            else:
                prefix = "In return, " if variant == 0 else "On the other hand, "
                clauses = [
                    _claim_text(c, prefer_state=prefer_state).rstrip(".")
                    for c in concessions
                ]
            # One sentence — a second bare concession would read as a merit.
            parts.append(prefix + " and ".join(clauses) + ".")

    alt = facts.better_alternative
    if alt is not None:
        alt_pv = _alt_pv_token(alt)
        if variant == 0:
            s = f"Better was {alt.san}"
            if alt.verdict:
                s += f", which {alt.verdict}"
            if alt_pv:
                s += f" after {alt_pv}"
        else:
            s = f"A better move was {alt.san}"
            if alt.verdict:
                s += f", {_gerundize(alt.verdict)}"
            if alt_pv:
                s += f" after {alt_pv}"
        s += "."
        if alt.claims:
            s += " " + " ".join(c.text for c in alt.claims)
        parts.append(s)
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# LLM rendering — one call, three audience registers
# ---------------------------------------------------------------------------

SINGLE_LEVEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
    },
    "required": ["text"],
    "additionalProperties": False,
}

GUID_COMPOSER_SYSTEM = (
    "You are a chess annotator. You are given INVIOLABLE FACTS about one move: "
    "a verdict, a variation token, an evaluation token, and positional claims "
    "produced by a rule-based expert system (each claim may come in a change-form "
    "and a state-form — use whichever reads naturally). Write ONE rendering of "
    "the annotation for the audience specified below.\n\n"
    "HARD RULES:\n"
    "- Copy the EVAL token and every PV token into the text VERBATIM, unchanged.\n"
    "- State the verdict and convey EVERY claim (rephrase fluently; merging "
    "related claims into one sentence is encouraged).\n"
    "- NEVER add a position-specific assertion that is not among the claims: no "
    "new squares, files, piece placements, threats, plans, tactics, or judgments "
    "about THIS position. General chess knowledge about a named concept (what a "
    "doubled pawn is, why the bishop pair usually matters) is allowed only where "
    "the audience rules below say so — and only about features named in claims.\n"
    "- Express the evaluation ONLY through the verdict words and the eval token; "
    "never convert centipawns into 'pawns up' language.\n"
    "- MOVER PERSPECTIVE: explain why the move serves the side that played it. "
    "Claims listed as MERITS are what the move achieves — lead with them. Claims "
    "listed as CONCESSIONS favor the opponent: phrase them strictly as trade-offs "
    "('in return', 'at the cost of') for sound moves, or as the move's drawbacks "
    "('now the opponent ...') when the concession mode says 'consequence'. NEVER "
    "present a concession as an achievement of the move.\n"
    "- If REFUTATION is present, the text MUST name that move as what punishes "
    "the played move (it is the board-level reason the move fails).\n"
    "- If BETTER ALTERNATIVE is present, end with one sentence naming it ('Better "
    "was {move}...' or a varied equivalent) with its verdict, claims and PV token.\n"
    "- One paragraph. No lists, no headers, no engine-worship.\n"
)

AUDIENCE_BLOCKS: Dict[str, str] = {
    "expert": (
        "AUDIENCE — expert: register of Chess Informant / grandmaster game "
        "collections. Dry, terse, declarative; ~25-55 words besides tokens. Do "
        "NOT explain features, concepts or terms — the reader is a strong "
        "player. No flavor, no narrative, no rhetorical questions. Merge the "
        "claims into compact compound sentences.\n"
    ),
    "intermediate": (
        "AUDIENCE — intermediate: club player. ~50-85 words besides tokens. "
        "Standard terms used, never defined. You MAY add one short clause on "
        "why the single most important claimed feature generally matters. "
        "Plain, practical tone.\n"
    ),
    "beginner": (
        "AUDIENCE — beginner: learning player. ~70-115 words besides tokens. "
        "Explain in a simple way each targeted positional feature (e.g. what a "
        "passed pawn is) the first time it is named; avoid jargon or define it "
        "inline; friendly instructive tone; you MAY close with one short "
        "general takeaway tied to a claimed concept. Never patronize and never "
        "invent new analysis.\n"
    ),
}

ENRICHMENT_RULES = (
    "ENRICHMENT block (optional): intermediate and beginner MAY weave in at most "
    "one element (opening background or what this leads to later in the game) as "
    "scene-setting; expert must ignore it entirely.\n"
)


def build_facts_user_prompt(
    facts: CommentFacts,
    *,
    enrichment: Optional[Dict[str, Any]] = None,
) -> str:
    def _claim_line(c: Claim) -> str:
        line = f"- {c.text}"
        if c.text_state:
            line += f" | state-form: {c.text_state}"
        if c.flag_note:
            line += f" [{c.flag_note}]"
        if c.features_involved:
            line += f" (features: {', '.join(c.features_involved[:3])})"
        return line

    merit_lines = [_claim_line(c) for c in facts.claims if not c.is_concession]
    concession_lines = [_claim_line(c) for c in facts.claims if c.is_concession]
    if not merit_lines and not concession_lines:
        merit_lines = ["- (no positional claims fired; comment on verdict and line only)"]

    blocks: List[str] = [
        "INVIOLABLE FACTS:",
        f"Move: {_move_label(facts)} (played by {facts.mover}, {facts.phase}game)",
        f"Verdict: this move {facts.verdict}",
        f"EVAL token (copy verbatim): {eval_token(facts)}",
        f"PV token (copy verbatim): {pv_token(facts)}",
    ]
    if facts.refutation_san:
        blocks.append(f"REFUTATION (must be named): {facts.refutation_san}")
    blocks += [
        f"MERITS (what the move achieves for {facts.mover}):",
        *(merit_lines or ["- (none)"]),
    ]
    if concession_lines:
        blocks += [
            f"CONCESSIONS (favor the opponent; mode: {facts.concession_mode}):",
            *concession_lines,
        ]
    alt = facts.better_alternative
    if alt is not None:
        alt_claims = [f"- {c.text}" for c in alt.claims] or ["- (none)"]
        blocks += [
            "",
            "BETTER ALTERNATIVE:",
            f"Move: {alt.san}",
            f"Verdict: {alt.verdict}",
            f"PV token (copy verbatim): {_alt_pv_token(alt)}",
            "Claims:",
            *alt_claims,
        ]
    if enrichment:
        enr_lines: List[str] = []
        for key in ("opening", "episode_theme", "what_happens_later", "master_note"):
            val = enrichment.get(key)
            if val:
                enr_lines.append(f"{key}: {val}")
        if enr_lines:
            blocks += ["", "ENRICHMENT (intermediate/beginner only, optional):", *enr_lines]
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Post-validation
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


_SCHEMA_MARKERS = ('additionalProperties', '"type":"object"', '"type": "object"', '"properties"')


def validate_facts_comment(text: str, facts: CommentFacts) -> bool:
    """The fact contract: eval and PV tokens verbatim; alternative named.

    Also rejects raw JSON / schema leakage so a malformed provider response can
    never reach the UI (it falls back to the deterministic template instead)."""
    t = _normalize(text)
    if not t:
        return False
    # Reject schema echoes and raw JSON objects outright.
    if t.lstrip().startswith('{') or any(m in t for m in _SCHEMA_MARKERS):
        return False
    ev = eval_token(facts)
    if ev and ev not in t:
        return False
    pv = pv_token(facts)
    if pv and _normalize(pv) not in t:
        return False
    alt = facts.better_alternative
    if alt is not None and alt.san and alt.san not in t:
        return False
    if facts.refutation_san and facts.refutation_san not in t:
        return False
    return True


_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")


def _parse_text(raw: str) -> str:
    """Extract the comment text from the provider response.

    Robust to providers that echo the JSON schema and/or concatenate several
    objects: scans every balanced top-level ``{...}`` and returns the ``text``
    of the LAST object that carries a non-empty, non-schema ``text`` field.
    """
    if not raw:
        return ""
    # Happy path: the whole response is the object.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("text"), str) and obj["text"].strip():
            return obj["text"].strip()
    except Exception:
        pass
    # Scan for embedded objects; prefer the last valid {text: ...}.
    best = ""
    for m in _OBJ_RE.finditer(raw):
        chunk = m.group(0)
        if "additionalProperties" in chunk or '"properties"' in chunk:
            continue  # this is the echoed schema, not a response
        try:
            o = json.loads(chunk)
        except Exception:
            continue
        if isinstance(o, dict) and isinstance(o.get("text"), str) and o["text"].strip():
            best = o["text"].strip()
    if best:
        return best
    # Last resort: strip schema-looking lines and surrounding braces.
    cleaned = "\n".join(
        ln for ln in raw.splitlines()
        if "additionalProperties" not in ln and '"type"' not in ln and '"properties"' not in ln
    ).strip()
    return cleaned


async def compose_facts_comment(
    service: Any,
    facts: CommentFacts,
    *,
    model: Optional[str],
    effort: str,
    enrichment: Optional[Dict[str, Any]] = None,
    level: str = DEFAULT_LEVEL,
) -> Dict[str, Any]:
    """Render the facts at the audience level chosen at submit time.

    Returns ``{"text": str, "rendering": "llm"|"template", "contract_ok": bool}``;
    an LLM text violating the fact contract falls back to the template.
    """
    lvl = level if level in LEVELS else DEFAULT_LEVEL
    template = render_facts_template(facts)

    configured = True
    try:
        configured = bool(service._provider.is_configured())
    except Exception:
        configured = False
    if not llm_rendering_enabled() or not configured:
        return {"text": template, "rendering": "template", "contract_ok": True, "level": lvl}

    system = GUID_COMPOSER_SYSTEM + "\n" + AUDIENCE_BLOCKS[lvl] + "\n" + ENRICHMENT_RULES
    user = build_facts_user_prompt(facts, enrichment=enrichment)
    candidate = ""
    try:
        raw = await service._llm_call_json_schema(
            system,
            user,
            model=model,
            effort=effort,
            schema=SINGLE_LEVEL_SCHEMA,
            schema_name="facts_comment",
            max_output_tokens=500,
        )
        candidate = _parse_text(raw)
    except Exception as e:
        logger.warning("facts composer LLM call failed (ply %s): %s", facts.ply, e)

    if candidate and validate_facts_comment(candidate, facts):
        return {"text": candidate, "rendering": "llm", "contract_ok": True, "level": lvl}
    if candidate:
        logger.info("facts comment failed contract at ply %s — using template", facts.ply)
    return {"text": template, "rendering": "template", "contract_ok": False, "level": lvl}
