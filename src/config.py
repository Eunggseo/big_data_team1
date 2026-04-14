import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL")


    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    DATA_DIR = os.getenv("DATA_DIR", "./data/docs")
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./storage/faiss_index")
    CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "./storage/cache.sqlite")

    TOP_K = int(os.getenv("TOP_K", "8"))
    TOP_N = int(os.getenv("TOP_N", "3"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

    RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.45"))
    RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.20"))


config = Config()