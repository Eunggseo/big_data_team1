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
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinicalRAG — MIMIC-IV",
    layout="wide",
    page_icon="⚕",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #F7F9FC;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #1A2035;
}

/* Kill Streamlit's default top padding */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 0px !important;
    padding-bottom: 0 !important;
    margin-top: -60px !important;
}

[data-testid="stHeader"] {
    display: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E4E9F2;
    padding: 0;
}

[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}

.sidebar-header {
    padding: 24px 20px 16px;
    border-bottom: 1px solid #E4E9F2;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
}

.sidebar-logo-icon {
    width: 32px;
    height: 32px;
    background: #0F2D6B;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 16px;
    flex-shrink: 0;
}

.sidebar-title {
    font-size: 15px;
    font-weight: 600;
    color: #1A2035;
    letter-spacing: -0.2px;
}

.sidebar-subtitle {
    font-size: 11px;
    color: #8A96B0;
    margin-top: 2px;
    letter-spacing: 0.3px;
}

.sidebar-section {
    padding: 16px 20px 8px;
}

.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8A96B0;
    margin-bottom: 10px;
}

/* History items */
.history-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 4px;
    border: 1px solid transparent;
    transition: all 0.15s ease;
}

.history-item:hover {
    background: #F0F4FF;
    border-color: #C7D7F7;
}

.history-item.active {
    background: #EEF3FF;
    border-color: #B8CEFF;
}

.history-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #0F2D6B;
    flex-shrink: 0;
    opacity: 0.6;
}

.history-text {
    font-size: 12px;
    color: #3D4A6B;
    line-height: 1.5;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.history-time {
    font-size: 10px;
    color: #A8B3CC;
    margin-top: 2px;
}

/* ── Main content area ── */
.main-container {
    max-width: 860px;
    margin: 0 auto;
    padding: 8px 24px 120px;
}

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}

.top-bar-title {
    font-size: 26px;
    font-weight: 600;
    color: #1A2035;
    letter-spacing: -0.5px;
}

.top-bar-badge {
    font-size: 13px;
    background: #EEF1F8;
    color: #0F2D6B;
    border: 1px solid #C2CCDF;
    border-radius: 20px;
    padding: 5px 14px;
    font-weight: 500;
}

/* ── Welcome state ── */
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px 32px;
}

.welcome-icon {
    width: 52px;
    height: 52px;
    background: #0F2D6B;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    margin-bottom: 20px;
    color: white;
}

.welcome-heading {
    font-size: 28px;
    font-weight: 600;
    color: #1A2035;
    letter-spacing: -0.6px;
    margin-bottom: 10px;
}

.welcome-subtext {
    font-size: 17px;
    color: #7A88A8;
    font-weight: 400;
    letter-spacing: -0.1px;
}

/* ── Chat messages ── */
.chat-scroll {
    margin-bottom: 24px;
}

/* User message bubble */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
}

.msg-user-bubble {
    background: #1A56DB;
    color: #FFFFFF;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    font-size: 16px;
    line-height: 1.6;
    max-width: 75%;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Assistant message */
.msg-assistant {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    align-items: flex-start;
}

.msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #EEF3FF;
    border: 1px solid #C7D7F7;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}

.msg-assistant-content {
    flex: 1;
}

/* Answer card */
.answer-card {
    background: #FFFFFF;
    border: 1px solid #E4E9F2;
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 16px;
    line-height: 1.75;
    color: #2A3350;
    margin-bottom: 10px;
    font-family: 'IBM Plex Sans', sans-serif;
    white-space: pre-line;
}

/* Meta row */
.meta-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}

.badge {
    font-size: 13px;
    border-radius: 20px;
    padding: 4px 12px;
    font-weight: 500;
    border: 1px solid;
    white-space: nowrap;
}

.badge-blue {
    background: #EEF1F8;
    color: #0F2D6B;
    border-color: #C2CCDF;
}

.badge-green {
    background: #EDFBF3;
    color: #0A7A3E;
    border-color: #AAEACC;
}

.badge-orange {
    background: #FFF5E6;
    color: #A05800;
    border-color: #FFD599;
}

.badge-gray {
    background: #F3F5FA;
    color: #6B7A9A;
    border-color: #D8E0F0;
}

/* Rewrite row */
.rewrite-row {
    font-size: 12px;
    color: #6B7A9A;
    font-style: italic;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.rewrite-arrow {
    color: #1A56DB;
    font-style: normal;
    font-size: 11px;
}

/* Sources */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 10px;
}

.source-pill {
    background: #F3F5FA;
    border: 1px solid #D8E0F0;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 11px;
    color: #4A5888;
    font-family: 'IBM Plex Mono', monospace;
}

/* Retrieved chunks expander */
.chunk-card {
    background: #F7F9FC;
    border: 1px solid #E4E9F2;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.chunk-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #0F2D6B;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-score {
    font-size: 11px;
    color: #8A96B0;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-meta {
    font-size: 11px;
    color: #A8B3CC;
    margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-content {
    font-size: 12px;
    color: #4A5888;
    line-height: 1.6;
}

/* ── Input bar ── */
.input-wrapper {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, #F7F9FC 85%, transparent);
    padding: 16px 24px 24px;
    z-index: 100;
}

.input-inner {
    max-width: 860px;
    margin: 0 auto;
}

/* Streamlit overrides */
[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    border: 1px solid #D0D9EE !important;
    border-radius: 12px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1A2035 !important;
    padding: 12px 16px !important;
    resize: none !important;
    box-shadow: 0 2px 8px rgba(26,86,219,0.06) !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #0F2D6B !important;
    box-shadow: 0 0 0 3px rgba(15,45,107,0.1) !important;
    outline: none !important;
}

/* Streamlit button tweaks */
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: #0F2D6B !important;
    border-color: #0F2D6B !important;
    color: white !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #0A2050 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(15,45,107,0.3) !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid #E4E9F2 !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary {
    font-size: 13px !important;
    color: #6B7A9A !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    width: 100%;
    text-align: left !important;
    background: transparent !important;
    border: 1px solid #E4E9F2 !important;
    color: #3D4A6B !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #F0F4FF !important;
    border-color: #C2CCDF !important;
    color: #0F2D6B !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D0D9EE; border-radius: 4px; }

/* Spinner */
[data-testid="stSpinner"] {
    color: #1A56DB !important;
}

/* Success / warning */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
}

/* Divider */
hr { border-color: #E4E9F2; }
</style>
""", unsafe_allow_html=True)

# ─── Init session state ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

init_cache()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / header
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">⚕</div>
            <div>
                <div class="sidebar-title">ClinicalRAG</div>
                <div class="sidebar-subtitle">MIMIC-IV · Discharge Notes</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Index management
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">Index</div>', unsafe_allow_html=True)
    if st.button("⟳  Build / Refresh FAISS Index", use_container_width=True):
        os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
        with st.spinner("Building index..."):
            build_index()
        st.success("Index built.")
    st.markdown(f'<div style="font-size:11px; color:#8A96B0; padding: 6px 4px;">📁 {config.DATA_DIR}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin: 4px 20px;">', unsafe_allow_html=True)

    # Chat history
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-label">History</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        user_turns = [
            (i, msg) for i, msg in enumerate(st.session_state.chat_history)
            if msg["role"] == "user"
        ]
        for idx, (i, msg) in enumerate(reversed(user_turns[-10:])):
            q = msg["content"]
            short = q[:60] + "…" if len(q) > 60 else q
            ts = msg.get("timestamp", "")
            st.markdown(f"""
            <div class="history-item">
                <div class="history-dot"></div>
                <div>
                    <div class="history-text">{short}</div>
                    <div class="history-time">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("", unsafe_allow_html=True)
        if st.button("✕  Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.markdown('<div style="font-size:12px; color:#A8B3CC; padding: 4px 4px 12px;">No queries yet.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Main content ────────────────────────────────────────────────────────────
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Top bar
n_queries = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-title">Clinical Insights</div>
    <span class="top-bar-badge">{n_queries} {'query' if n_queries == 1 else 'queries'} this session</span>
</div>
""", unsafe_allow_html=True)

# ── Welcome / Empty state ─────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">⚕</div>
        <div class="welcome-subtext">Surface clinical patterns from patient notes.</div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat messages ─────────────────────────────────────────────────────────────
else:
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-user-bubble">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

        elif msg["role"] == "assistant":
            result = msg["result"]
            n_docs = len(result.reranked_docs) if result.reranked_docs else 0

            # Badges
            badges = f"<span class='badge badge-blue'>📄 {n_docs} notes</span>"
            if result.cache_hit:
                badges += "<span class='badge badge-gray'>⚡ cached</span>"
            if result.used_fallback:
                badges += "<span class='badge badge-orange'>⚠ fallback</span>"

            # Sources
            sources_html = ""
            if result.reranked_docs:
                sources = list(dict.fromkeys([
                    d["metadata"].get("source", "Unknown")
                    for d in result.reranked_docs if d.get("metadata")
                ]))
                pills = "".join([f"<span class='source-pill'>{s}</span>" for s in sources])
                sources_html = f'<div class="sources-row">{pills}</div>'

            # Rewrite
            rewrite_html = ""
            query_used = msg.get("query", "")
            if result.rewritten_query and result.rewritten_query.strip() != query_used.strip():
                rewrite_html = f"""
                <div class="rewrite-row">
                    <span class="rewrite-arrow">↪</span>
                    Rewritten: <em>{result.rewritten_query}</em>
                </div>
                """

            st.markdown(f"""
            <div class="msg-assistant">
                <div class="msg-avatar">⚕</div>
                <div class="msg-assistant-content">
                    <div class="meta-row">{badges}</div>
                    {rewrite_html}
                    <div class="answer-card">{result.answer}</div>
                    {sources_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Chunks expander
            if result.reranked_docs:
                with st.expander(f"📂 Retrieved chunks ({n_docs})", expanded=False):
                    for i, doc in enumerate(result.reranked_docs, start=1):
                        meta = doc.get("metadata", {})
                        note_id = meta.get("note_id") or meta.get("source") or f"Doc {i}"
                        rerank_score = doc.get("rerank_score")
                        retrieval_score = doc.get("score")
                        score_str = ""
                        if rerank_score is not None:
                            score_str += f"Rerank {rerank_score:.3f}"
                        if retrieval_score is not None:
                            score_str += f"  ·  Retrieval {retrieval_score:.3f}"

                        st.markdown(f"""
                        <div class="chunk-card">
                            <div class="chunk-header">
                                <span class="chunk-tag">chunk {i} · {note_id}</span>
                                <span class="chunk-score">{score_str}</span>
                            </div>
                            <div class="chunk-meta">{meta}</div>
                            <div class="chunk-content">{doc.get("content", "")[:400]}{'…' if len(doc.get("content","")) > 400 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Debug expander
            with st.expander("🛠 Debug", expanded=False):
                st.json({
                    "cache_hit": result.cache_hit,
                    "rewritten_query": result.rewritten_query,
                    "retry_count": result.retry_count,
                    "used_fallback": result.used_fallback,
                    "retrieved_docs": [
                        {"source": d["metadata"].get("source"), "note_id": d["metadata"].get("note_id"), "score": d.get("score")}
                        for d in result.retrieved_docs
                    ],
                    "reranked_docs": [
                        {"source": d["metadata"].get("source"), "note_id": d["metadata"].get("note_id"), "rerank_score": d.get("rerank_score")}
                        for d in result.reranked_docs
                    ]
                })

    st.markdown('</div>', unsafe_allow_html=True)

# ── Input area (pinned bottom) ────────────────────────────────────────────────
st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)  # spacer

st.markdown('<div class="input-wrapper"><div class="input-inner">', unsafe_allow_html=True)

input_col, btn_col = st.columns([10, 1])

with input_col:
    query = st.text_area(
        label="query_input",
        label_visibility="collapsed",
        placeholder="Ask about patient treatments, diagnoses, rare patterns…",
        height=72,
        key="main_query_input"
    )

with btn_col:
    st.markdown('<div style="padding-top: 8px;">', unsafe_allow_html=True)
    run_clicked = st.button("→", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_clicked and query and query.strip():
    ts = time.strftime("%H:%M")

    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query.strip(),
        "timestamp": ts
    })

    with st.spinner("Retrieving clinical context…"):
        result = run_pipeline(query.strip())

    # Add assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "result": result,
        "query": query.strip()
    })

    st.rerun()

elif run_clicked and (not query or not query.strip()):
    st.warning("Please enter a query before running.")