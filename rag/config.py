import os

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 120
ROWS_PER_CHUNK = 15     # table rows per chunk
TOP_K_RETRIEVE = 20     # stage 1: FAISS candidates
TOP_K_FINAL = 5         # stage 2: after reranking

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | ollama
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
