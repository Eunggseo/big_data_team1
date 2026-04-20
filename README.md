# big_data_team1

# Clinical Discharge Note Intelligence System

> This project repository is created in partial fulfillment of the requirements 
> for the Big Data Analytics course offered by the Master of Science in Business 
> Analytics program at the Carlson School of Management, University of Minnesota.

---

## Project Overview

A RAG-based conversational AI system that enables clinicians and analysts to 
query medical discharge notes using natural language, powered by GPT-4 and 
AWS infrastructure.

---

## Use Cases

- **Use Case 1:** Find patients with similar clinical profiles (age, condition, 
  lab values) and retrieve relevant treatment patterns
- **Use Case 2:** Identify co-occurrence of specific drugs and symptoms across 
  patient populations

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


