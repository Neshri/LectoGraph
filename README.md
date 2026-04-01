# LectoGraph

Batch-ingest lecture videos into a [LightRAG](https://github.com/HKUDS/LightRAG) knowledge graph for cross-video querying.

Videos are transcribed with Whisper and analyzed frame-by-frame with a vision model via Ollama. The resulting structured documents are inserted into a persistent knowledge graph. Once built, the graph can be queried with natural language across all ingested lectures.

Processing state is tracked in SQLite — videos already ingested are skipped automatically, so you can cancel at any time and resume safely.

---

## Requirements

- Python 3.10+
- CUDA GPU (for Whisper transcription)
- [Ollama](https://ollama.com) with the following models pulled:
  ```bash
  ollama pull qwen3:32b
  ollama pull qwen3-embedding:8b
  ollama pull glm-ocr
  ```
- FFmpeg on PATH
- On first run, Whisper downloads from HuggingFace (~3 GB, cached after that)

```bash
pip install -r requirements.txt
```

---

## Getting started

1. Place video files in `input/`  
   Supported: `.mp4 .mkv .avi .mov .webm`

2. Review `config.yaml` — especially `frame_analysis_prompt` at the bottom, which should reflect your video content type

3. Check what would be processed:
   ```bash
   python ingest.py --dry-run
   ```

4. Test with one video first:
   ```bash
   python ingest.py --limit 1
   ```

5. Run the full batch:
   ```bash
   python ingest.py
   ```

---

## Ingestion (`ingest.py`)

```bash
python ingest.py                   # Process all pending videos
python ingest.py --status          # Show which videos are done / pending / failed
python ingest.py --dry-run         # Preview without processing
python ingest.py --limit N         # Process at most N videos this run
python ingest.py --retry-failed    # Re-queue failed videos, then process
python ingest.py --reset-db        # Wipe state DB and start fresh (graph is kept)
python ingest.py --config other.yaml
```

**Ctrl+C is safe** — the current video finishes and is committed before the process exits. Re-running picks up where it left off.

For long runs, use `tmux` so the process survives if your SSH session drops:

```bash
tmux new -s lectograph
python ingest.py

# Detach:   Ctrl+B, D
# Reattach: tmux attach -t lectograph

# Watch the log from another terminal:
tail -f logs/ingest_*.log
```

---

## Querying (`query.py`)

```bash
python query.py "Hur öppnar man kontrollpanelen i Windows 7?"

# --mode: hybrid (default), local, global, naive
python query.py --mode global "Vilka kurser behandlar nätverkssäkerhet?"

# --top-k: how many graph nodes to retrieve (default 20)
python query.py --top-k 30 "Vad är skillnaden mellan NTFS och FAT32?"

# --strict: appends an instruction to only answer from lesson content
python query.py --strict "Hur konfigurerar man en statisk IP-adress?"
```

---

## Configuration (`config.yaml`)

All settings in one file — no code changes needed to switch models or paths.
Relative paths resolve relative to the config file, not the working directory.

```yaml
input_folder: ./input        # Source videos
working_dir: ./knowledge_db  # LightRAG graph database
docs_dir: ./docs             # Saved knowledge documents
logs_dir: ./logs             # One log file per run

whisper_model: KBLab/kb-whisper-large
whisper_device: cuda
frame_analysis_model: glm-ocr
summary_model: qwen3:32b

ollama_url: http://127.0.0.1:11434
rag_llm_model: qwen3:32b
rag_embedding_model: qwen3-embedding:8b  # ⚠ Do not change after DB is created
rag_embedding_dim: 4096                  # Must match the embedding model
```

> **⚠ Embedding model warning:** Changing `rag_embedding_model` or `rag_embedding_dim` after the database exists will silently corrupt queries. If you need to switch models, delete `knowledge_db/` and re-ingest from scratch.

---

## Output

| Path | Contents |
|---|---|
| `knowledge_db/` | LightRAG graph, vector index, KV stores |
| `knowledge_db/ingestion_state.db` | Processing state per video |
| `docs/<name>_ingested.txt` | Knowledge document generated for each video |
| `logs/ingest_<timestamp>.log` | Full log for each run |

---

## Project structure

```
LectoGraph/
├── config.yaml          ← Edit this to configure everything
├── ingest.py            ← Ingestion CLI
├── query.py             ← Query CLI
├── lectograph/
│   ├── config.py        ← Config dataclass
│   ├── state.py         ← SQLite state tracker
│   └── pipeline.py      ← Ingestion loop + document formatter
├── input/               ← Put video files here
├── knowledge_db/        ← Generated: LightRAG database
├── docs/                ← Generated: knowledge documents
└── logs/                ← Generated: run logs
```