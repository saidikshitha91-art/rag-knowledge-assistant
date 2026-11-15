"""Ties every stage together: upload -> chunk -> embed -> FAISS -> retrieve -> rerank -> LLM."""
from collections import defaultdict

from . import analysis, config, llm
from .chunker import split_text
from .embeddings import Embedder
from .loaders import Segment, load_docx, load_pdf, load_tabular, table_to_segments
from .reranker import Reranker
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.reranker = Reranker()
        self.store = None
        self.tables = {}                      # (filename, sheet) -> DataFrame
        self.chunks_by_file = defaultdict(list)

    # ---------- ingestion ----------
    def ingest(self, name: str, data: bytes) -> int:
        lower = name.lower()
        segments = []
        if lower.endswith(".pdf"):
            segments = self._chunk_segments(load_pdf(data, name))
        elif lower.endswith(".docx"):
            segments = self._chunk_segments(load_docx(data, name))
        elif lower.endswith((".csv", ".xlsx", ".xls")):
            for sheet, df in load_tabular(data, name).items():
                self.tables[(name, str(sheet))] = df
                segments += table_to_segments(df, name, str(sheet), config.ROWS_PER_CHUNK)
        else:
            raise ValueError(f"Unsupported file type: {name}")

        if not segments:
            return 0
        vecs = self.embedder.encode([s.text for s in segments])
        if self.store is None:
            self.store = VectorStore(vecs.shape[1])
        records = [{"text": s.text, "source": s.source, "location": s.location} for s in segments]
        self.store.add(vecs, records)
        self.chunks_by_file[name].extend(records)
        return len(segments)

    @staticmethod
    def _chunk_segments(segs):
        out = []
        for seg in segs:
            for piece in split_text(seg.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                out.append(Segment(piece, seg.source, seg.location))
        return out

    # ---------- question answering ----------
    def ask(self, question: str):
        if not self.store or len(self.store) == 0:
            return "Please upload and index a document first.", []
        qvec = self.embedder.encode([question])[0]
        candidates = self.store.search(qvec, config.TOP_K_RETRIEVE)          # stage 1
        top = self.reranker.rerank(question, candidates, config.TOP_K_FINAL)  # stage 2
        return llm.answer_question(question, top), top

    # ---------- summarisation / insights ----------
    def summarize(self, filename: str) -> str:
        chunks = [c["text"] for c in self.chunks_by_file.get(filename, [])]
        if not chunks:
            return "No content found for this file."
        groups, buf = [], ""
        for c in chunks:
            if len(buf) + len(c) > 6000:
                groups.append(buf)
                buf = ""
            buf += c + "\n"
        groups.append(buf)
        groups = groups[:3]  # cap cost for very large files
        partials = [llm.generate("Summarize the passage concisely in bullet points.", g, 500)
                    for g in groups]
        if len(partials) == 1:
            return partials[0]
        return llm.generate(
            "Merge these partial summaries into one coherent summary with a short "
            "overview followed by key points.", "\n\n".join(partials), 900)

    def insights(self, key) -> str:
        df = self.tables[key]
        return llm.generate(
            "You are a data analyst. From the dataset profile, list key insights, "
            "notable patterns, data-quality issues, and suggested next analyses. Be specific.",
            analysis.profile_text(df), 1000)

    def anomalies(self, key, contamination: float = 0.05):
        return analysis.detect_anomalies(self.tables[key], contamination)
