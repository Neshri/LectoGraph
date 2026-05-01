# LectoGraph - Agent Instructions
Welcome to the LectoGraph codebase. This file provides context and guidelines to help you write accurate, maintainable, and stylistically consistent code for this project.
## Project Overview
LectoGraph batch-ingests lecture videos into a [LightRAG](https://github.com/HKUDS/LightRAG) knowledge graph, enabling cross-video semantic querying of the course material. 
It uses a multi-modal pipeline: vision processing (`glm-ocr` for frame analysis), audio transcription (`faster-whisper`), and structured synthesis (`qwen3:32b`).
## System Architecture
*   **Ingestion Pipeline (`ingest.py`, `lectograph/pipeline.py`)**: Asynchronously processes videos through `OpenSceneSense`, formats the output into a knowledge document (to maximize entity extraction), and inserts it into LightRAG.
*   **Query Pipeline (`query.py`)**: Interfaces with LightRAG to perform semantic searches (modes: `hybrid`, `local`, `global`, `naive`).
*   **State Management (`lectograph/state.py`)**: Uses SQLite (`ingestion_state.db`) to track processing states (`pending`, `analyzing`, `ingested`, `failed`). This guarantees safe resume-ability if the process is killed.
*   **Configuration (`config.yaml`, `lectograph/config.py`)**: A single source of truth for all paths, models, hyperparameters, and prompts.
## Tech Stack
*   **Language**: Python 3.10+
*   **Core Libraries**: LightRAG, `faster-whisper`, `asyncio`, `pathlib`, `sqlite3`
*   **AI Engine**: Local models served via [Ollama](https://ollama.com)
---
## Coding Guidelines
### 1. Types and Syntax
*   **Modern Typing**: Always use `from __future__ import annotations` at the top of files. Use modern Python 3.10+ type hints (e.g., `list[str]`, `dict[str, Any]`, `Path | None` instead of `Union` or `Optional`).
*   **Paths**: Always use `pathlib.Path`. Never use `os.path`.
### 2. Configuration-Driven Development
*   Never hardcode model names, file paths, parameters, or prompts in the Python code.
*   All new configurable values must be added to `config.yaml` and mapped in the `Config` dataclass in `lectograph/config.py`.
*   *Note*: Relative paths in `config.yaml` are strictly resolved relative to the config file's directory, **not** the current working directory (`CWD`).
### 3. Asynchronous & Safe Execution
*   **Async First**: LightRAG operations (`ainsert`, `adelete_by_doc_id`, etc.) are async. Use `asyncio` for pipeline integrations.
*   **Graceful Shutdowns**: The ingestion loop must remain resumable. Respect the `stop_event` (a `threading.Event`) to finish processing the *current* video before shutting down. Do not kill processes mid-flight if possible.
*   **State Tracking**: If a step fails, log the error and mark the file as `failed` in the SQLite DB so it can be retried via `--retry-failed`.
### 4. Knowledge Graph Best Practices
*   **Document Formatting**: The text fed into LightRAG (`format_knowledge_doc`) must be dense, factual prose. Do not include operational metadata (like frame counts or timestamps) as they create junk graph triples and dilute extraction quality.
*   **Immutable Embeddings**: The `rag_embedding_model` and `rag_embedding_dim` cannot be changed after the database is created without wiping the DB entirely. Treat these as immutable in production code.
### 5. AI Hallucination & Patching Rules
*   If adding new hallucination checks, implement them cleanly in `lectograph/pipeline.py` (e.g., `_is_faulty()`), keeping regex patterns compiled globally to ensure high performance.