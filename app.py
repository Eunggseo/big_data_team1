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

/*
 * Notion AI-inspired monochrome palette
 * --bg-main:    #FAFAFA  near-white clinical background
 * --bg-sidebar: #F4F4F2  slightly deeper sidebar
 * --bg-card:    #FFFFFF  pure white for elevated cards
 * --bg-input:   #F1F1EF  slightly deeper input zone
 * --border:     #E8E8E5  hairline borders
 * --text-1:     #1A1A1A  near-black primary text
 * --text-2:     #52525B  secondary labels
 * --text-3:     #A1A1AA  muted/meta
 * --text-4:     #D4D4D8  ghost text
 */

.stApp {
    background: #FAFAFA;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #1A1A1A;
}

/* Kill Streamlit's default top padding */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 10px !important;
    padding-bottom: 0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #F4F4F2;
    border-right: 1px solid #E8E8E5;
    padding: 0;
    min-width: 300px !important;
    max-width: 300px !important;
}

/* Force sidebar visible even if Streamlit remembered a collapsed state */
[data-testid="stSidebar"][aria-expanded="false"] {
    min-width: 300px !important;
    max-width: 300px !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
    visibility: visible !important;
}

[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}

.sidebar-header {
    padding: 24px 20px 16px;
    border-bottom: 1px solid #E8E8E5;
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
    background: #3F3F46;
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
    color: #1A1A1A;
    letter-spacing: -0.2px;
}

.sidebar-subtitle {
    font-size: 11px;
    color: #A1A1AA;
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
    color: #A1A1AA;
    margin-bottom: 10px;
}

/* History items */
.history-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 6px;
    cursor: pointer;
    margin-bottom: 2px;
    border: 1px solid transparent;
    transition: all 0.12s ease;
}

.history-item:hover {
    background: #EBEBEA;
    border-color: #E0E0DE;
}

.history-item.active {
    background: #EBEBEA;
    border-color: #D4D4D2;
}

.history-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #52525B;
    flex-shrink: 0;
    opacity: 0.5;
    margin-top: 5px;
}

.history-text {
    font-size: 12px;
    color: #52525B;
    line-height: 1.5;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}

.history-time {
    font-size: 10px;
    color: #A1A1AA;
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
    color: #1A1A1A;
    letter-spacing: -0.5px;
}

.top-bar-badge {
    font-size: 12px;
    background: #F4F4F2;
    color: #52525B;
    border: 1px solid #E8E8E5;
    border-radius: 20px;
    padding: 4px 12px;
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
    width: 48px;
    height: 48px;
    background: #F4F4F2;
    border: 1px solid #E8E8E5;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 20px;
    color: #52525B;
}

.welcome-heading {
    font-size: 26px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.6px;
    margin-bottom: 10px;
}

.welcome-subtext {
    font-size: 15px;
    color: #A1A1AA;
    font-weight: 400;
    letter-spacing: -0.1px;
}

/* ── Chat messages ── */
.chat-scroll {
    margin-bottom: 24px;
}

/* User message bubble — light gray, consistent with sidebar */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 16px;
}

.msg-user-bubble {
    background: #EBEBEA;
    color: #1A1A1A;
    border: 1px solid #E0E0DE;
    padding: 11px 17px;
    border-radius: 16px 16px 4px 16px;
    font-size: 15px;
    line-height: 1.6;
    max-width: 72%;
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
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: #F4F4F2;
    border: 1px solid #E8E8E5;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
    margin-top: 2px;
    color: #52525B;
}

.msg-assistant-content {
    flex: 1;
}

/* Answer card — pure white, elevated */
.answer-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E5;
    border-radius: 10px;
    padding: 18px 22px;
    font-size: 15px;
    line-height: 1.78;
    color: #1A1A1A;
    margin-bottom: 10px;
    font-family: 'IBM Plex Sans', sans-serif;
    white-space: pre-line;
    word-wrap: break-word;
    overflow-wrap: break-word;
    overflow-x: hidden;
}

/* Meta row */
.meta-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}

.badge {
    font-size: 11px;
    border-radius: 4px;
    padding: 3px 10px;
    font-weight: 500;
    border: 1px solid;
    white-space: nowrap;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.1px;
}

/* All badges use the same neutral monochrome system */
.badge-blue {
    background: #F4F4F2;
    color: #3F3F46;
    border-color: #E4E4E7;
}

.badge-green {
    background: #F0FDF4;
    color: #166534;
    border-color: #BBF7D0;
}

.badge-orange {
    background: #FFF7ED;
    color: #9A3412;
    border-color: #FED7AA;
}

.badge-gray {
    background: #F4F4F2;
    color: #71717A;
    border-color: #E4E4E7;
}

/* Rewrite row */
.rewrite-row {
    font-size: 11.5px;
    color: #71717A;
    font-style: italic;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.rewrite-arrow {
    color: #3F3F46;
    font-style: normal;
    font-size: 11px;
    opacity: 0.7;
}

/* Sources */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 10px;
}

.source-pill {
    background: #F4F4F2;
    border: 1px solid #E4E4E7;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10.5px;
    color: #71717A;
    font-family: 'IBM Plex Mono', monospace;
}

/* Retrieved chunks expander */
.chunk-card {
    background: #FAFAFA;
    border: 1px solid #E8E8E5;
    border-radius: 7px;
    padding: 11px 14px;
    margin-bottom: 6px;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}

.chunk-tag {
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #3F3F46;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-score {
    font-size: 10.5px;
    color: #A1A1AA;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-meta {
    font-size: 10.5px;
    color: #A1A1AA;
    margin-bottom: 5px;
    font-family: 'IBM Plex Mono', monospace;
}

.chunk-content {
    font-size: 12px;
    color: #52525B;
    line-height: 1.65;
}

/* ── Input bar ── */
.input-wrapper {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, #F1F1EF 82%, transparent);
    padding: 16px 24px 24px;
    z-index: 100;
}

.input-inner {
    max-width: 860px;
    margin: 0 auto;
}

/* Target both text_input and text_area since Streamlit version may vary */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    border: 1px solid #E0E0DE !important;
    border-radius: 10px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1A1A1A !important;
    padding: 11px 15px !important;
    resize: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #1A1A1A !important;
    box-shadow: 0 0 0 2px rgba(26,26,26,0.08) !important;
    outline: none !important;
}

/* Streamlit button tweaks */
[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.12s !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: #EBEBEA !important;
    border-color: #D4D4D2 !important;
    color: #3F3F46 !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #E0E0DE !important;
    border-color: #C8C8C6 !important;
    color: #1A1A1A !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07) !important;
}

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 300px !important;
    max-width: 300px !important;
}

[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid #E8E8E5 !important;
    border-radius: 7px !important;
}

[data-testid="stExpander"] summary {
    font-size: 12.5px !important;
    color: #71717A !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    width: 100%;
    text-align: left !important;
    background: transparent !important;
    border: 1px solid #E8E8E5 !important;
    color: #52525B !important;
    padding: 7px 11px !important;
    border-radius: 6px !important;
    font-size: 12.5px !important;
}

[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
    background: #EBEBEA !important;
    border-color: #D4D4D2 !important;
    color: #1A1A1A !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D4D4D8; border-radius: 3px; }

/* Spinner */
[data-testid="stSpinner"] {
    color: #52525B !important;
}

/* Success / warning */
[data-testid="stAlert"] {
    border-radius: 7px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important;
}

/* Divider */
hr { border-color: #E8E8E5; }

/* ── Evidence Vault — right column panel ── */

/* Wrapper gives the column a left-border separator and consistent padding */
.ev-col-wrapper {
    border-left: 1px solid #E8E8E5;
    min-height: 75vh;
    padding: 0 4px 140px 16px;
}

/* Vault header block */
.ev-panel-hdr {
    padding: 2px 0 12px 0;
    border-bottom: 1px solid #E8E8E5;
    margin-bottom: 12px;
}

.ev-panel-lbl {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #A1A1AA;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 3px;
}

.ev-panel-title {
    font-size: 13px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.2px;
}

/* Evidence cards list */
.ev-cards-wrap {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

/* Individual evidence card */
.ev-card {
    border: 1px solid #E8E8E5;
    border-radius: 8px;
    overflow: hidden;
    background: #FFFFFF;
    transition: border-color 0.15s, box-shadow 0.15s;
}

.ev-card:hover {
    border-color: #D4D4D2;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
}

/* Low-confidence card: amber left accent border */
.ev-rare {
    border-left: 3px solid #FED7AA;
}

/* Card header area */
.ev-head {
    padding: 8px 11px 7px;
    background: #FAFAFA;
    border-bottom: 1px solid #F0F0EE;
}

.ev-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2px;
}

/* Note ID in monospace */
.ev-noteid {
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    color: #3F3F46;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 65%;
}

/* Rerank score — color-coded by tier */
.ev-score-h { font-size: 10px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: #166534; }
.ev-score-m { font-size: 10px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: #52525B; }
.ev-score-l { font-size: 10px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: #9A3412; }

/* Source type + rare flag row */
.ev-meta2 {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 5px;
    flex-wrap: wrap;
}

.ev-srctype {
    font-size: 9.5px;
    color: #A1A1AA;
    font-family: 'IBM Plex Mono', monospace;
}

/* RARE flag badge for low-confidence docs */
.ev-rare-flag {
    font-size: 8.5px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #9A3412;
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 3px;
    padding: 0 4px;
    line-height: 1.6;
}

/* Retrieval score pushed to right */
.ev-ret {
    font-size: 9px;
    color: #A1A1AA;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: auto;
}

/* 2px score progress bar */
.ev-score-bar {
    height: 2px;
    background: #E8E8E5;
    border-radius: 2px;
    overflow: hidden;
    margin-top: 5px;
}

.ev-score-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s ease; }
.ev-score-bar-fill.h { background: #86EFAC; }  /* green  — high confidence */
.ev-score-bar-fill.m { background: #CBD5E1; }  /* slate  — medium          */
.ev-score-bar-fill.l { background: #FCA5A5; }  /* red-ish — low            */

/* Confidence badge below score bar */
.ev-conf {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    border: 1px solid;
    margin-top: 5px;
    letter-spacing: 0.3px;
}
.ev-conf.h { background: #F0FDF4; color: #166534; border-color: #BBF7D0; }
.ev-conf.m { background: #F4F4F2; color: #52525B;  border-color: #E4E4E7; }
.ev-conf.l { background: #FFF7ED; color: #9A3412;  border-color: #FED7AA; }

/* Card body: content snippet */
.ev-body { padding: 8px 11px 10px; }

.ev-snippet {
    font-size: 11.5px;
    color: #52525B;
    line-height: 1.65;
    word-break: break-word;
}

/* Empty / waiting state */
.ev-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 48px 16px;
    border: 1px solid #E8E8E5;
    border-radius: 8px;
    background: #FFFFFF;
}

.ev-empty-icon {
    font-size: 26px;
    opacity: 0.12;
    margin-bottom: 12px;
}

.ev-empty-txt {
    font-size: 12px;
    color: #A1A1AA;
    line-height: 1.75;
    max-width: 185px;
}
</style>
""", unsafe_allow_html=True)

# ─── Init session state ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Evidence Vault state — holds docs from the most recent pipeline run
if "ev_docs" not in st.session_state:
    st.session_state.ev_docs = []
if "ev_query" not in st.session_state:
    st.session_state.ev_query = ""

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
    st.markdown(f'<div style="font-size:11px; color:#A1A1AA; padding: 6px 4px;">📁 {config.DATA_DIR}</div></div>', unsafe_allow_html=True)

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
            st.session_state.ev_docs  = []
            st.session_state.ev_query = ""
            st.rerun()
    else:
        st.markdown('<div style="font-size:12px; color:#A1A1AA; padding: 4px 4px 12px;">No queries yet.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── Main content — two-column layout ────────────────────────────────────────
# Left  [5]: chat thread + pinned input bar
# Right [3]: Evidence Vault — shows reranked docs from the latest query
col_chat, col_vault = st.columns([5, 3], gap="small")

# ── LEFT: Chat column ─────────────────────────────────────────────────────────
with col_chat:
    st.markdown('<div style="padding: 0 8px 0 24px;">', unsafe_allow_html=True)

    # Top bar — title, query counter, and index/clear actions
    n_queries = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-title">Clinical Insights</div>
        <span class="top-bar-badge">{n_queries} {'query' if n_queries == 1 else 'queries'} this session</span>
    </div>
    """, unsafe_allow_html=True)

    # Welcome / empty state
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">⚕</div>
            <div class="welcome-subtext">Surface clinical patterns from patient notes</div>
        </div>
        """, unsafe_allow_html=True)

    # Chat message thread
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
                # For cache hits reranked_docs may be empty — fall back to retrieved_docs count
                docs_for_count = result.reranked_docs or result.retrieved_docs or []
                n_docs = len(docs_for_count)

                # Compact meta badges
                badges = f"<span class='badge badge-blue'>📄 {n_docs} notes</span>"
                if result.cache_hit:
                    badges += "<span class='badge badge-gray'>⚡ cached</span>"
                if result.used_fallback:
                    badges += "<span class='badge badge-orange'>⚠ fallback</span>"

                # Source pills — compact row (full chunk detail lives in Vault)
                docs_with_meta = result.reranked_docs or result.retrieved_docs or []
                sources_html = ""
                if docs_with_meta:
                    sources = list(dict.fromkeys([
                        d["metadata"].get("source", "Unknown")
                        for d in docs_with_meta if d.get("metadata")
                    ]))
                    pills = "".join([f"<span class='source-pill'>{s}</span>" for s in sources])
                    sources_html = f'<div class="sources-row">{pills}</div>'

                st.markdown(f"""
                <div class="msg-assistant">
                    <div class="msg-avatar">⚕</div>
                    <div class="msg-assistant-content">
                        <div class="meta-row">{badges}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Answer rendered as Streamlit markdown, indented past the avatar
                st.markdown('<div style="margin-left: 40px;">', unsafe_allow_html=True)
                with st.container():
                    st.markdown(result.answer)
                if sources_html:
                    st.markdown(sources_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Debug expander (chunk details moved to Evidence Vault)
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

    # Spacer so content isn't hidden behind the pinned input bar
    st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── RIGHT: Evidence Vault column ───────────────────────────────────────────────
with col_vault:
    st.markdown('<div class="ev-col-wrapper">', unsafe_allow_html=True)

    ev_docs  = st.session_state.get("ev_docs", [])
    ev_query = st.session_state.get("ev_query", "")

    # ── Score classification helpers ──────────────────────────────────────────
    def _score_tier(s):
        """Map a rerank score to a confidence tier: 'h', 'm', or 'l'."""
        if s is None:
            return "m"
        return "h" if s >= 0.7 else ("m" if s >= 0.4 else "l")

    def _score_label(tier):
        return {"h": "HIGH", "m": "MED", "l": "LOW"}[tier]

    def _bar_pct(s):
        """Convert a 0–1 score to a CSS percentage width string."""
        if s is None:
            return "40%"
        return f"{min(int(s * 100), 100)}%"

    # Vault header
    n_ev = len(ev_docs)
    query_short = (ev_query[:42] + "…") if len(ev_query) > 42 else ev_query
    count_line = (
        "<span style='color:#A1A1AA;font-family:IBM Plex Mono,monospace;font-size:10px;'>No query yet</span>"
        if not ev_docs else
        f"<span style='font-family:IBM Plex Mono,monospace;font-size:10px;color:#71717A;'>"
        f"{n_ev} chunk{'s' if n_ev != 1 else ''}</span>"
        f"<span style='font-size:10px;color:#A1A1AA;font-style:italic;'> · {query_short}</span>"
    )
    st.markdown(f"""
    <div class="ev-panel-hdr">
        <div class="ev-panel-lbl">Evidence Vault</div>
        <div class="ev-panel-title">Retrieved Notes</div>
        <div style="margin-top:5px;">{count_line}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ev-cards-wrap">', unsafe_allow_html=True)

    if not ev_docs:
        # Waiting-state placeholder
        st.markdown("""
        <div class="ev-empty">
            <div class="ev-empty-icon">📋</div>
            <div class="ev-empty-txt">Submit a clinical query to see retrieved note chunks here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for i, doc in enumerate(ev_docs, start=1):
            meta         = doc.get("metadata", {})
            note_id      = meta.get("note_id") or meta.get("source") or f"doc_{i}"
            src_type     = meta.get("category") or meta.get("note_type") or "discharge note"
            rerank_score = doc.get("rerank_score")
            ret_score    = doc.get("score")
            tier         = _score_tier(rerank_score)
            rare_cls     = "ev-rare" if tier == "l" else ""
            rare_flag    = '<span class="ev-rare-flag">RARE</span>' if tier == "l" else ""
            ret_str      = f"ret {ret_score:.3f}" if ret_score is not None else ""
            score_val    = f"{rerank_score:.3f}" if rerank_score is not None else "—"
            bar_pct      = _bar_pct(rerank_score)
            content      = doc.get("content", "")
            snippet      = content[:220] + ("…" if len(content) > 220 else "")

            st.markdown(f"""
            <div class="ev-card {rare_cls}">
                <div class="ev-head">
                    <div class="ev-top">
                        <span class="ev-noteid">{note_id}</span>
                        <span class="ev-score-{tier}">{score_val}</span>
                    </div>
                    <div class="ev-meta2">
                        <span class="ev-srctype">{src_type}</span>
                        {rare_flag}
                        <span class="ev-ret">{ret_str}</span>
                    </div>
                    <div class="ev-score-bar">
                        <div class="ev-score-bar-fill {tier}" style="width:{bar_pct}"></div>
                    </div>
                    <span class="ev-conf {tier}">● {_score_label(tier)}</span>
                </div>
                <div class="ev-body">
                    <div class="ev-snippet">{snippet}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # ev-cards-wrap
    st.markdown('</div>', unsafe_allow_html=True)  # ev-col-wrapper

# ── Input area (pinned bottom — fixed position spans full viewport width) ─────
st.markdown('<div class="input-wrapper"><div class="input-inner">', unsafe_allow_html=True)

input_col, btn_col = st.columns([10, 1])

with input_col:
    query = st.text_input(
        label="query_input",
        label_visibility="collapsed",
        placeholder="Ask about patient treatments, diagnoses, rare patterns…",
        key="main_query_input"
    )

with btn_col:
    run_clicked = st.button("→", type="primary", use_container_width=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# ── Run pipeline ───────────────────────────────────────────────────────────────
should_run = run_clicked or (
    query and query.strip() and
    st.session_state.get("last_submitted") != query.strip()
)

if should_run and query and query.strip():
    st.session_state["last_submitted"] = query.strip()
    ts = time.strftime("%H:%M")

    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query.strip(),
        "timestamp": ts
    })

    with st.spinner("Retrieving clinical context…"):
        result = run_pipeline(query.strip())

    # Add assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "result": result,
        "query": query.strip()
    })

    # Push latest retrieved docs to the Evidence Vault
    st.session_state["ev_docs"]  = result.reranked_docs or result.retrieved_docs or []
    st.session_state["ev_query"] = query.strip()

    st.rerun()

elif run_clicked and (not query or not query.strip()):
    st.warning("Please enter a query before running.")
