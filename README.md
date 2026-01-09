# RAGx

End-to-end Retrieval-Augmented Generation pipeline built with LangChain, ChromaDB, and Google Gemini.

## Features

- Document ingestion (PDF, TXT, Markdown)
- Semantic chunking with configurable parameters
- Vector storage with ChromaDB (local, persistent)
- Question answering powered by Gemini 3 Flash
- Interactive CLI with Rich formatting
- Rate limit handling with automatic retry
- FastAPI backend with REST endpoints

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

### Installation

```bash
git clone https://github.com/hiharshit/ragx.git
cd ragx
uv sync

cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Usage

```bash
# Ingest documents
uv run ragx ingest data/

# Ask a question
uv run ragx query "What is this document about?"

# Start interactive chat
uv run ragx chat

# View stats
uv run ragx stats

# Clear vector store
uv run ragx clear
# Clear vector store
uv run ragx clear

# Start API server
uv run uvicorn api.main:app --reload
```

### API Documentation

Once the server is running, you can access the interactive API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
ragx/
├── src/ragx/              # Core library
│   ├── config.py          # Settings management
│   ├── embeddings.py      # Gemini embeddings
│   ├── vectorstore.py     # ChromaDB operations
│   ├── document_loader.py # PDF & text loaders
│   ├── chunker.py         # Text splitting
│   ├── llm.py             # Gemini LLM
│   ├── retriever.py       # Vector retrieval
│   └── rag_chain.py       # Main RAG pipeline
├── cli/                   # Command-line interface
│   └── main.py
├── api/                   # FastAPI routes
│   ├── routes/            # API endpoints
│   ├── main.py
│   └── schemas.py
├── frontend/              # Web UI (coming soon)
├── data/                  # Document storage
├── pyproject.toml         # UV/Python config
└── .env.example           # Environment template
```

## Configuration

Environment variables (`.env`):

| Variable         | Description                      |
| ---------------- | -------------------------------- |
| `GEMINI_API_KEY` | Google Gemini API key (required) |

Application settings (`src/ragx/config.py`):

| Setting           | Default                       |
| ----------------- | ----------------------------- |
| `gemini_model`    | `gemini-3-flash-preview`      |
| `embedding_model` | `models/gemini-embedding-001` |
| `chunk_size`      | `1000`                        |
| `chunk_overlap`   | `200`                         |

## Tech Stack

- **LangChain** - RAG orchestration
- **ChromaDB** - Vector storage
- **Google Gemini** - LLM & embeddings
- **FastAPI** - REST API
- **Typer + Rich** - CLI interface
- **Pydantic Settings** - Configuration

## License

MIT
