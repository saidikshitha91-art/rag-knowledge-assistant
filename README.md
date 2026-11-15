# AI-Powered Multi-Domain Knowledge and Insights Assistant (RAG)

Pipeline: upload -> extract/clean -> chunk -> BGE-M3 embeddings -> FAISS -> top-20 retrieval
-> cross-encoder rerank (top-5) -> LLM grounded answer. Plus summarization, insights, anomaly detection.

## Run
    python -m venv venv
    venv\Scripts\activate            (Windows)   |   source venv/bin/activate  (Mac/Linux)
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your-key   (Windows)   |   export ANTHROPIC_API_KEY=your-key
    streamlit run app.py

Low RAM (8 GB)? Set EMBED_MODEL=BAAI/bge-small-en-v1.5 and
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 before running.
No API key? Install Ollama, run `ollama pull llama3.1`, set LLM_PROVIDER=ollama.
