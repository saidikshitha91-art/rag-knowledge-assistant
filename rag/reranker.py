"""Step 7: stage-2 reranking with a cross-encoder."""
from sentence_transformers import CrossEncoder

from . import config


class Reranker:
    def __init__(self, model_name: str = config.RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list, top_k: int):
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
