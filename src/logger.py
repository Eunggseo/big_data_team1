import os
import json
from datetime import datetime
from src.config import config


LOG_DIR = "./logs"


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_run(state):
    ensure_log_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"log_{timestamp}.json"
    path = os.path.join(LOG_DIR, filename)

    log_data = {
        "query": state.query,
        "rewritten_query": state.rewritten_query,
        "cache_hit": state.cache_hit,
        "retry_count": state.retry_count,
        "used_fallback": state.used_fallback,
        "answer": state.answer,

        "retrieved_docs": [
            {
                "score": d.get("score"),
                "metadata": d.get("metadata"),
                "preview": d.get("content", "")[:300]
            }
            for d in state.retrieved_docs
        ],

        "reranked_docs": [
            {
                "rerank_score": d.get("rerank_score"),
                "metadata": d.get("metadata"),
                "preview": d.get("content", "")[:300]
            }
            for d in state.reranked_docs
        ],

        "debug": state.debug
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"Log saved: {path}")