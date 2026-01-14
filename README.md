# AI Research Paper Helper - Backend

A FastAPI-based backend providing ML-powered summarization, equation explanation, and RAG-based Q&A for research papers.

## Requirements

- Python 3.9+
- CUDA-compatible GPU (optional, for faster inference)

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /health` - Health check
- `POST /summarize` - Generate multi-level summaries
- `POST /explain-equations` - Explain LaTeX equations
- `POST /extract-key-points` - Extract key contributions and concepts
- `POST /rag/index` - Index a paper for RAG queries
- `POST /rag/query` - Query indexed paper with natural language

## Configuration

Set environment variables or modify `config.py`:

- `API_MODE` - `local`, `api`, or `hybrid`
- `OPENROUTER_API_KEY` - For external LLM API (optional)
- `MODEL_CACHE_DIR` - Directory for model caching
