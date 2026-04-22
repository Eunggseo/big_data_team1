import sqlite3
import pandas as pd
from src.config import config

df = pd.read_csv(
    config.DATA_PATH,
    usecols=["note_id", "subject_id", "hadm_id", "text", "charttime"],
    nrows=config.MAX_ROWS_LOAD
)

conn = sqlite3.connect(config.SQLITE_DB_PATH)

print("[CSV_TO_SQLITE] Writing SQLite database...")
df.to_sql("notes", conn, if_exists="replace", index=False)

conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON notes(subject_id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_note ON notes(note_id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_hadm ON notes(hadm_id)")

conn.commit()
conn.close()

print("[CSV_TO_SQLITE] SQLite built.")