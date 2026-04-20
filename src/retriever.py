from typing import List, Dict, Any
from src.vectorstore import load_faiss_index
from src.config import config


def retrieve(query: str, k: int | None = None) -> List[Dict[str, Any]]:
    k = k or config.TOP_K
    db = load_faiss_index()

    # STAGE 1: Embedding / Similarity Search
    # Find the mathematical matches
    initial_results = db.similarity_search_with_score(query, k=k)

    if not initial_results:
        return []


    # Extract unique note_ids from best matching chunks
    # Use a set() so if multiple hits come from the same note, we don't duplicate it
    target_note_ids = set()
    for doc, score in initial_results:
        note_id = doc.metadata.get("note_id")
        if note_id:
            target_note_ids.add(note_id)

    # STAGE 2: Structured / Metadata Search
    # Go into FAISS doc store and grab ALL chunks belonging to those notes
    expanded_docs = []
    
    # db.docstore._dict holds all the raw Document objects we saved during ingest.py
    for doc_id, doc in db.docstore._dict.items():
        if doc.metadata.get("note_id") in target_note_ids:
            expanded_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 0.0, # We force score to 0.0 since these were pulled by ID, not by similarity math
            })

    # Fallback to initial results if no metadata found
    if not expanded_docs:
        for doc, score in initial_results:
            expanded_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            })

    return expanded_docs

    # results = db.similarity_search_with_score(query, k=k)

    # docs = []
    # for doc, score in initial_results:
    #     docs.append({
    #         "content": doc.page_content,
    #         "metadata": doc.metadata,
    #         "score": float(score),
    #     })
    # return docs