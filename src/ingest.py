import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import config
from src.vectorstore import save_faiss_index
from src.data_sources.csv_loader import CSVLoader


def load_documents(data_dir: str) -> List[Document]:
    docs: List[Document] = []

    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())

    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # tune
        chunk_overlap=150, # tune 
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")

    return chunks




def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


# def build_index():
#     loader = CSVLoader(
#         path="/Users/mashhoodkhan/Downloads/trends_data/discharge-001.csv",
#         nrows=10000,
#         filters={}  # e.g. {"subject_id": 12345}
#     )

#     docs = loader.load()
#     chunks = chunk_documents(docs)

#     save_faiss_index(chunks)

#     print(f"Indexed {len(chunks)} chunks.")

import os
import pickle
from src.data_sources.csv_loader import CSVLoader
from src.embeddings import get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.config import config


CHECKPOINT_PATH = "./storage/embedding_checkpoint.pkl"


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


def build_index():
    loader = CSVLoader(
        path="/Users/mashhoodkhan/Downloads/trends_data/discharge-001.csv",
        nrows=100, # running into issues with usage limits, testing with 100 rows
        filters={}
    )

    docs = loader.load()
    chunks = chunk_documents(docs)

    embedding_model = get_embedding_model()

    # -------- LOAD CHECKPOINT --------
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            saved = pickle.load(f)
            embedded_chunks = saved["chunks"]
            start_idx = saved["idx"]
    else:
        embedded_chunks = []
        start_idx = 0

    print(f"Resuming from chunk {start_idx}")

    batch_size = 50  # tune this

    for i in range(start_idx, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]

        embeddings = embedding_model.embed_documents(texts)

        for text, meta, emb in zip(texts, metadatas, embeddings):
            embedded_chunks.append((text, meta, emb))

        # -------- SAVE CHECKPOINT --------
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump({
                "chunks": embedded_chunks,
                "idx": i + batch_size
            }, f)

        print(f"Processed {i + batch_size}/{len(chunks)}")

    # -------- BUILD FAISS --------
    texts = [x[0] for x in embedded_chunks]
    metas = [x[1] for x in embedded_chunks]
    embs = [x[2] for x in embedded_chunks]

    db = FAISS.from_embeddings(
    text_embeddings=list(zip(texts, embs)),
    embedding=embedding_model,
    metadatas=metas
    )

    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    db.save_local(config.FAISS_INDEX_DIR)

    print("Index built successfully.")