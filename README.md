# big_data_team1

# Clinical Note Intelligence with RAG
**Helping Clinicians Find Relevant Answers Faster from Unstructured Notes**

> This project repository is created in partial fulfillment of the requirements 
> for the Big Data Analytics course offered by the Master of Science in Business 
> Analytics program at the Carlson School of Management, University of Minnesota.

---

## Executive Summary

Clinical notes are stored as unstructured text that is long, fragmented, and inconsistent, making it difficult for clinicians to effectively search across patients or answer critical questions quickly. 

This project introduces a Retrieval-Augmented Generation (RAG) system that combines structured and semantic search with LLM generation to solve this challenge. By using a FAISS vector database for semantic similarity and GPT-4 for grounded generation, the Streamlit web interface reduces hours of manual chart review into seconds, providing evidence-backed clinical decision-making at scale.

---

## Use Cases

- **Patient-Level Analysis (Chart Review):** Deep dive into a specific patient's history. Quickly extract treatments, responses, and diagnoses from unstructured medical notes to accelerate individual chart review.
- **Population-Level Insights (Scalable Analysis):** Identify broad clinical patterns across patient groups. Find patients with similar clinical profiles (e.g., age, condition, lab values) and surface trends like drug-symptom co-occurrences to support evidence-based clinical decision-making at scale.

---

## Key System Capabilities
* **Intent-Aware Routing:** Uses an agentic AI query parser to route queries across semantic search, patient search, note search, and visit search. 
* **Context Preservation:** Retrieves parent documents to preserve context, expanding beyond standard isolated chunk retrieval.
* **Grounded Generation & Citations:** GPT-4 synthesizes answers strictly bounded by the retrieved evidence, completely reducing unsupported responses (hallucinations).
* **Confidence Scoring:** Every response generated always includes note-level citations and confidence indicators from supporting notes.

---

## System Architecture

```
User
↓
Streamlit UI
↓
HTTPS Endpoint
↓
API Service (AWS ECS/Fargate)
↓
RAG Orchestrator
- agent logic · query rewrite · retrieval · reranker · cache
↓
OpenAI API
- embeddings API → convert query to vector
- generation API → produce final answer from retrieved context
↓
Vector Store (FAISS) + Cache/Database
↓
Response returned to UI
```

---

## Repository Structure

```
src/
├── data_sources/      # Raw and processed data references
├── agent.py           # Agent logic and orchestration
├── cache.py           # Query caching layer
├── config.py          # Environment and model configuration
├── embeddings.py      # Text embedding via OpenAI API
├── evaluator.py       # RAGAS-based evaluation pipeline
├── ingest.py          # Data ingestion and chunking (FAISS cosine similarity)
├── llm.py             # GPT-4 generation calls
├── logger.py          # Logging utilities
├── pipeline.py        # End-to-end RAG pipeline
├── reranker.py        # Chunk reranking logic
├── retriever.py       # Vector store retrieval
├── state.py           # Session/state management
└── vectorstore.py     # FAISS vector store setup and query
```

---

## Dataset

- **MIMIC-IV 3.1** — Clinical discharge notes (mimic-iv-note-2.2)
- `discharge.csv` — joinable to `admissions.csv` on `hadm_id`
- `chunks.parquet` (1.73 GB) — Section-level chunked notes with metadata 
  (note_id, subject_id, hadm_id)
- `RAG_evaluation_dataset_new.xlsx` — 20 clinical QA pairs for evaluation
- All data stored on shared AWS S3 bucket

---

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Eunggseo/big_data_team1.git
cd big_data_team1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export OPENAI_API_KEY=your_openai_key
export AWS_ACCESS_KEY_ID=your_aws_key
export AWS_SECRET_ACCESS_KEY=your_aws_secret

# 4. Ingest data into vector store
python src/ingest.py

# 5. Run the Streamlit UI
streamlit run app.py
```

---

## Related Resources
- 📊 [Dataset — MIMIC-IV](https://physionet.org/content/mimiciv/3.1/)


## Team Members (Team 1)
* Ethan Armstrong
* Ziqi Cao
* Ko-Jung Hsu
* Cole Johnson
* Mashhood Khan
* Wenyu Zhong

