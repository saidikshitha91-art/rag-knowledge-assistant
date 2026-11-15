import pandas as pd
import streamlit as st

from rag import analysis
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="Multi-Domain Knowledge & Insights Assistant", layout="wide")
st.title("AI-Powered Multi-Domain Knowledge and Insights Assistant")
st.caption("RAG over PDF, Word, Excel and CSV files — BGE-M3 + FAISS + two-stage reranking + LLM")


@st.cache_resource(show_spinner="Loading embedding and reranker models (first run downloads them)...")
def get_pipeline():
    return RAGPipeline()


pipe = get_pipeline()
if "indexed" not in st.session_state:
    st.session_state.indexed = set()

with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader("PDF / DOCX / XLSX / CSV", type=["pdf", "docx", "xlsx", "xls", "csv"],
                             accept_multiple_files=True)
    if st.button("Index files", type="primary") and files:
        for f in files:
            if f.name in st.session_state.indexed:
                continue
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    n = pipe.ingest(f.name, f.getvalue())
                    st.session_state.indexed.add(f.name)
                    st.success(f"{f.name}: {n} chunks")
                except Exception as e:
                    st.error(f"{f.name}: {e}")
    st.write("**Indexed:**", ", ".join(sorted(st.session_state.indexed)) or "none yet")

tab_qa, tab_sum, tab_data = st.tabs(["Ask questions", "Summarize documents", "Data insights & anomalies"])

with tab_qa:
    q = st.text_input("Ask something about your documents")
    if st.button("Ask") and q:
        with st.spinner("Retrieving, reranking, generating..."):
            answer, sources = pipe.ask(q)
        st.markdown(answer)
        if sources:
            with st.expander("Source passages used"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {s['source']} — {s['location']}** "
                                f"(similarity {s['score']:.2f}, rerank {s['rerank_score']:.2f})")
                    st.text(s["text"])

with tab_sum:
    docs = [n for n in pipe.chunks_by_file if n in st.session_state.indexed]
    if docs:
        choice = st.selectbox("Choose a file", docs)
        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                st.markdown(pipe.summarize(choice))
    else:
        st.info("Index a file first.")

with tab_data:
    keys = list(pipe.tables.keys())
    if keys:
        key = st.selectbox("Choose a dataset", keys, format_func=lambda k: f"{k[0]} / {k[1]}")
        df = pipe.tables[key]
        st.dataframe(df.head(50), use_container_width=True)
        st.subheader("Column profile")
        st.dataframe(analysis.profile(df), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Generate insights"):
                with st.spinner("Analyzing..."):
                    st.markdown(pipe.insights(key))
        with c2:
            cont = st.slider("Expected anomaly fraction", 0.01, 0.20, 0.05, 0.01)
            if st.button("Detect anomalies"):
                result, msg = pipe.anomalies(key, cont)
                st.write(msg)
                if not result.empty:
                    st.dataframe(result, use_container_width=True)
    else:
        st.info("Upload an Excel or CSV file to enable this tab.")
