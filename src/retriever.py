from typing import List, Dict, Any
from src.vectorstore import load_faiss_index
from src.config import config


def retrieve(query: str, k: int | None = None) -> List[Dict[str, Any]]:
    k = k or config.TOP_K
    db = load_faiss_index()

    results = db.similarity_search_with_score(query, k=k)

    docs = []
    for doc, score in results:
        docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        })
    return docs