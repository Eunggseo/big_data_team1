import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
sys.stderr = open(os.devnull, 'w')

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from src.pipeline import run_pipeline
from src.ingest import build_index
from src.cache import init_cache
from src.config import config

st.set_page_config(page_title="RAG App", layout="wide")
st.title("Patient Notes Chatbot")

init_cache()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Indexing")
    st.write(f"Docs folder: `{config.DATA_DIR}`")
    if st.button("Build / Refresh FAISS Index"):
        os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
        build_index()
        st.success("Index built successfully.")

with col2:
    st.subheader("Ask")
    query = st.text_area("Enter query", height=120)

    if st.button("Run RAG") and query.strip():
        with st.spinner("Running pipeline..."):
            result = run_pipeline(query)

        st.markdown("### Answer")
        st.write(result.answer)

        with st.expander("Debug"):
            st.json({
                "cache_hit": result.cache_hit,
                "rewritten_query": result.rewritten_query,
                "retry_count": result.retry_count,
                "used_fallback": result.used_fallback,
                "retrieved_docs": [
                    {
                        "source": d["metadata"].get("source"),
                        "score": d.get("score"),
                    }
                    for d in result.retrieved_docs
                ],
                "reranked_docs": [
                    {
                        "source": d["metadata"].get("source"),
                        "rerank_score": d.get("rerank_score"),
                    }
                    for d in result.reranked_docs
                ]
            })

        with st.expander("Top Contexts"):
            for i, doc in enumerate(result.reranked_docs, start=1):
                st.markdown(f"**Doc {i}**")
                st.write(doc["content"])
                st.caption(str(doc["metadata"]))
                st.markdown("---")