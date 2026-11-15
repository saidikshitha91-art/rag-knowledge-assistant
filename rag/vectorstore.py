"""Step 5: FAISS index."""
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)   # exact cosine search (vectors are normalised)
        self.records = []                     # parallel list of {"text","source","location"}

    def add(self, vectors: np.ndarray, records: list):
        self.index.add(vectors)
        self.records.extend(records)

    def search(self, qvec: np.ndarray, k: int):
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(qvec.reshape(1, -1), k)
        return [dict(self.records[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def __len__(self):
        return self.index.ntotal
