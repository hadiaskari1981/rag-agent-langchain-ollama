# Local RAG with Qdrant & Ollama

## Overview
This repository demonstrates a **local-first semantic search / RAG pipeline** using:
- **Qdrant** as a vector database (running locally via Docker)
- **Ollama** for local embeddings and text generation
- A Jupyter notebook (`rag_agent_1.ipynb`) that walks through ingestion, embedding, storage, and querying

The goal is to provide a fully offline, reproducible setup suitable for experimentation, prototyping, and learning.

---

## Repository Structure
```
.
├── rag_agent_1.ipynb   # Main notebook: ingestion, embedding, search, generation
├── README.md              # This file
```

---

## Prerequisites

- Docker (Docker Desktop or Docker Engine)
- Python 3.9+
- pip / virtualenv (or conda)

---

## 1. Running Qdrant Locally with Docker

Pull and run Qdrant:

```bash
docker pull qdrant/qdrant

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verify Qdrant is running:
- REST API: http://localhost:6333
- Web UI: http://localhost:6333/dashboard

To stop and remove:
```bash
docker stop qdrant && docker rm qdrant
```

---

## 2. Running Ollama Locally

### Install Ollama

Follow the official installation instructions for your OS:
- macOS / Linux / Windows (WSL supported)

Once installed, verify:
```bash
ollama --version
```

### Pull Required Models

#### Embedding model
```bash
ollama pull nomic-embed-text
```

#### Text generation model (example)
```bash
ollama pull llama3
```

You can replace `llama3` with any other supported model depending on your hardware.

### Start Ollama

Ollama runs as a local service automatically. By default:
- API endpoint: `http://localhost:11434`

Test embeddings:
```bash
ollama run nomic-embed-text "This is a test sentence"
```

Test generation:
```bash
ollama run llama3 "Explain vector databases in one paragraph."
```

---

## 3. Python Environment Setup

Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies (example):
```bash
pip install qdrant-client requests jupyter numpy
```

Launch Jupyter:
```bash
jupyter notebook
```

---

## 4. Notebook Walkthrough (`semantic_search.ipynb`)

The notebook is structured as follows:

1. **Environment & Imports**
   - Python dependencies
   - Configuration for Qdrant and Ollama endpoints

2. **Embedding Function**
   - Calls Ollama embedding API
   - Transforms text into dense vectors

3. **Qdrant Collection Setup**
   - Create or reuse a collection
   - Define vector size and distance metric

4. **Data Ingestion**
   - Prepare documents
   - Embed and upsert into Qdrant

5. **Semantic Search**
   - Embed query
   - Retrieve nearest neighbors from Qdrant

6. **Optional Generation (RAG-style)**
   - Pass retrieved context to Ollama LLM
   - Generate a grounded response

---

## 5. Notes & Best Practices

- Embeddings are **required** for semantic search; Qdrant stores vectors, not raw text.
- Keep embedding and query models consistent.
- For production, consider:
  - Persistent volumes for Qdrant
  - Batch upserts
  - Metadata filtering

---

## License

MIT (or adapt as needed)

