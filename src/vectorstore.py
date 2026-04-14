import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embeddings import get_embedding_model
from src.config import config


def save_faiss_index(documents: List[Document]) -> None:
    embeddings = get_embedding_model()
    db = FAISS.from_documents(documents, embeddings)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    db.save_local(config.FAISS_INDEX_DIR)


def load_faiss_index() -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )