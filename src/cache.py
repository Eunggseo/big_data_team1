import sqlite3
from typing import Optional
from src.config import config
import os



def init_cache() -> None:
    os.makedirs(os.path.dirname(config.CACHE_DB_PATH), exist_ok=True)

    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            query TEXT PRIMARY KEY,
            response TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_cached_response(query: str) -> Optional[str]:
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT response FROM cache WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def store_response(query: str, response: str) -> None:
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO cache (query, response)
        VALUES (?, ?)
    """, (query, response))
    conn.commit()
    conn.close()


def clear_response_cache() -> None:
    init_cache()
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM cache")
    conn.commit()
    conn.close()
