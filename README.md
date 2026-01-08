# RAGx

End-to-end Retrieval-Augmented Generation pipeline built with LangChain, ChromaDB, and Google Gemini.

## Features

- Document ingestion (PDF, TXT, Markdown)
- Semantic chunking with configurable parameters
- Vector storage with ChromaDB (local, persistent)
- Question answering powered by Gemini 3 Flash
- Interactive CLI with Rich formatting

## Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

### Installation

```bash
# Clone the repository
git clone https://github.com/hiharshit/ragx.git
cd ragx

# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
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
```

## Project Structure

```
ragx/
├── src/ragx/           # Core library
│   ├── config.py       # Settings management
│   ├── embeddings.py   # Gemini embeddings
│   ├── vectorstore.py  # ChromaDB operations
│   ├── document_loader.py
│   ├── chunker.py
│   ├── llm.py          # Gemini LLM
│   ├── retriever.py
│   └── rag_chain.py    # Main pipeline
├── cli/                # Command-line interface
├── api/                # FastAPI routes (coming soon)
├── frontend/           # Web UI (coming soon)
└── data/               # Document storage
```

## Configuration

Environment variables (`.env`):

| Variable         | Description           | Default  |
| ---------------- | --------------------- | -------- |
| `GEMINI_API_KEY` | Google Gemini API key | Required |

Application settings (`src/ragx/config.py`):

| Setting           | Description            | Default                       |
| ----------------- | ---------------------- | ----------------------------- |
| `gemini_model`    | LLM model              | `gemini-3-flash-preview`      |
| `embedding_model` | Embedding model        | `models/gemini-embedding-001` |
| `chunk_size`      | Characters per chunk   | `1000`                        |
| `chunk_overlap`   | Overlap between chunks | `200`                         |

## Tech Stack

- **LangChain** - RAG orchestration
- **ChromaDB** - Vector storage
- **Google Gemini** - LLM & embeddings
- **Typer + Rich** - CLI interface
- **Pydantic Settings** - Configuration

## License

MIT
