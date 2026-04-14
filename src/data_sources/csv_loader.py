import pandas as pd
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document


class CSVLoader:
    def __init__(
        self,
        path: str,
        nrows: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ):
        self.path = path
        self.nrows = nrows
        self.filters = filters or {}

    def load(self) -> List[Document]:
        df = pd.read_csv(self.path, nrows=self.nrows)

        # ---- FILTER HERE ----
        for col, val in self.filters.items():
            df = df[df[col] == val]

        docs = []
        for _, row in df.iterrows():
            if not isinstance(row["text"], str):
                continue

            docs.append(
                Document(
                    page_content=row["text"],
                    metadata={
                        "note_id": row["note_id"],
                        "subject_id": row["subject_id"],
                        "hadm_id": row["hadm_id"],
                        "note_type": row["note_type"],
                        "charttime": row["charttime"],
                    },
                )
            )

        return docs