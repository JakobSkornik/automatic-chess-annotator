import asyncio
import chess
from app.core.commentary.rag_retriever import RAGQuery
from app.core.commentary.tantivy_positional_retriever import TantivyPositionalRetriever
r = TantivyPositionalRetriever("data/bm25_positions")
# Pick any FEN + PV (e.g. a Sicilian middlegame)
q = RAGQuery(
    fen="r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1B3/PPPQ1PPP/R3KB1R w KQ - 0 9",
    pv_san=["f3", "a6", "O-O-O", "Bd7", "g4"],
)
hits = asyncio.run(r.retrieve(q, top_k=5))
for h in hits:
    print(f"{h.similarity_score:.3f} {h.source} {h.fen[:40]}  {h.annotation_text[:60]}")