"""RAG retrieval interface for master-game annotations (stub default)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.chess_events import Episode, MoveEvent


class RAGQuery(BaseModel):
    """Structured query built from MoveEvent features."""

    pawn_structure_type: Optional[str] = None
    phase: Optional[str] = None
    eco: Optional[str] = None
    opening_name: Optional[str] = None
    ply: Optional[int] = None
    eco_prefix: Optional[str] = None
    tactical_motifs: List[str] = Field(default_factory=list)
    material_imbalance: Optional[str] = None
    eval_swing_direction: Optional[str] = None
    theme_hint: Optional[str] = None
    # BM25 / Tantivy: position after the move + PV SAN from that position (matches corpus build)
    fen: Optional[str] = None
    pv_san: Optional[List[str]] = None


class RAGResult(BaseModel):
    """A single retrieved annotated example."""

    source: str
    fen: Optional[str] = None
    annotation_text: str
    relevance_tags: Dict[str, str] = Field(default_factory=dict)
    similarity_score: Optional[float] = None


def _classify_imbalance(material: Optional[Dict[str, Any]]) -> Optional[str]:
    if not material or not isinstance(material, dict):
        return None
    diff = material.get("diff")
    if not isinstance(diff, dict):
        return None
    imb = material.get("imbalance")
    if isinstance(imb, list) and imb:
        return ",".join(str(x) for x in imb[:3])
    total = diff.get("total")
    if isinstance(total, (int, float)):
        if abs(total) < 50:
            return "equal"
        return "material_imbalance"
    return None


def build_rag_query(
    move_event: MoveEvent,
    episode: Optional[Episode] = None,
) -> RAGQuery:
    """Build a structured RAG query from a MoveEvent."""
    eco_full = (move_event.opening_eco or "").strip()
    eco_prefix = eco_full[:2] if len(eco_full) >= 2 else (eco_full if eco_full else None)

    swing_dir: Optional[str] = None
    if move_event.eval_swing_cp is not None:
        s = move_event.eval_swing_cp
        if s < -80:
            swing_dir = "swing_negative_white_pov"
        elif s > 80:
            swing_dir = "swing_positive_white_pov"

    return RAGQuery(
        pawn_structure_type=move_event.pawn_structure_type,
        phase=move_event.phase,
        eco=eco_full or None,
        opening_name=move_event.opening_name,
        ply=move_event.ply,
        eco_prefix=eco_prefix,
        tactical_motifs=[m.value for m in move_event.tactical_motifs],
        material_imbalance=_classify_imbalance(move_event.material_balance),
        eval_swing_direction=swing_dir,
        theme_hint=episode.dominant_theme if episode else None,
        fen=move_event.fen_after,
        pv_san=move_event.pv_san,
    )


class RAGRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: RAGQuery, top_k: int = 2) -> List[RAGResult]:
        ...


def rag_results_to_ws_refs(results: List[RAGResult]) -> List[Dict[str, Any]]:
    """Serialize RAG results for WebSocket `data.rag_refs` (FE + debugging)."""
    out: List[Dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "source": r.source,
                "fen": r.fen or "",
                "text": r.annotation_text[:300],
                "score": float(r.similarity_score) if r.similarity_score is not None else 0.0,
                "san": (r.relevance_tags.get("san") or ""),
                "phase": (r.relevance_tags.get("phase") or ""),
            }
        )
    return out
