#!/usr/bin/env python3
"""
Pipeline for ingesting video analysis into LightRAG.
Uses OpenSceneSense's cleaned summaries + transcript rather than raw frame timeline.
"""

import sys
import asyncio
import logging
import shutil
import numpy as np
from pathlib import Path

try:
    from openscenesense_ollama.models import AnalysisPrompts
    from openscenesense_ollama.transcriber import WhisperTranscriber
    from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
    from openscenesense_ollama.frame_selectors import DynamicFrameSelector

    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure you are running this in an environment with openscenesense-ollama and lightrag-hku installed.")
    sys.exit(1)


# ─── Configuration ───────────────────────────────────────────────────────────

VIDEO_PATH = "windows7kontrollpanel.mp4"

# OpenSceneSense Models
WHISPER_MODEL = "KBLab/kb-whisper-large"
FRAME_ANALYSIS_MODEL = "glm-ocr"
SUMMARY_MODEL = "qwen3:32b"

# LightRAG Models & Storage
OLLAMA_URL = "http://127.0.0.1:11434"
RAG_LLM_MODEL = "qwen3:32b"
RAG_EMBEDDING_MODEL = "qwen3-embedding:8b"
RAG_EMBEDDING_DIM = 4096
RAG_WORKING_DIR = "./test_course_knowledge_db"

TEST_QUERY = "Hur öppnar man kontrollpanelen och vilka inställningar syns på skärmen?"


# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("VideoToRAG")


# ─── Helper Functions ────────────────────────────────────────────────────────

def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


# ─── Main Async Pipeline  ────────────────────────────────────────────────────

async def run_pipeline():
    print_section("Step 1: Analyzing Video")

    custom_prompts = AnalysisPrompts(
        frame_analysis=(
            "Du analyserar en skärmbild från en IT-lektion. "
            "Beskriv endast vad som är direkt synligt: öppna fönster, programnamn, "
            "terminalutskrifter, fil- och mappstrukturer, dialogrutor och text på skärmen. "
            "Dra inga slutsatser eller antaganden om sådant som inte syns direkt."
        )
    )

    transcriber = WhisperTranscriber(
        model_name=WHISPER_MODEL,
        device="cuda"
    )

    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=FRAME_ANALYSIS_MODEL,
        summary_model=SUMMARY_MODEL,
        min_frames=5,
        max_frames=45,
        frames_per_minute=3.0,
        frame_selector=DynamicFrameSelector(threshold=70.0),
        audio_transcriber=transcriber,
        prompts=custom_prompts,
        request_timeout=600.0,
        request_retries=1,
        log_level=logging.INFO
    )

    logger.info(f"Extracting data from {VIDEO_PATH}...")

    try:
        results = analyzer.analyze_video_structured(VIDEO_PATH)
    except Exception as e:
        logger.error(f"Video analysis failed: {e}", exc_info=True)
        return


    print_section("Step 2: Formatting Knowledge Document")

    filename = Path(VIDEO_PATH).name
    knowledge_doc = f"""# Lektion: {filename}
Längd: {results.metadata.video_duration:.1f} sekunder
Antal skärmutdrag: {results.metadata.num_frames_analyzed}
Antal ljudsegment: {results.metadata.num_audio_segments}

## Vad som sades (Transkription)
{results.summary.transcript}

## Detaljerad genomgång
{results.summary.detailed}

## Kort sammanfattning
{results.summary.brief}
"""

    logger.info("Generated Document Preview:")
    print("-" * 40)
    print(knowledge_doc[:1000] + "\n\n...[dokument trunkerat för översikt]...")
    print("-" * 40)

    doc_path = "latest_ingested_doc.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(knowledge_doc)
    logger.info(f"Full document saved to {doc_path}")


    print_section("Step 3: Initializing LightRAG Database")

    if Path(RAG_WORKING_DIR).exists():
        logger.info(f"Clearing old LightRAG storage at {RAG_WORKING_DIR}")
        shutil.rmtree(RAG_WORKING_DIR)
    Path(RAG_WORKING_DIR).mkdir(parents=True, exist_ok=True)

    @wrap_embedding_func_with_attrs(
        embedding_dim=RAG_EMBEDDING_DIM,
        max_token_size=8192,
        model_name=RAG_EMBEDDING_MODEL,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await ollama_embed.func(
            texts,
            embed_model=RAG_EMBEDDING_MODEL,
            host=OLLAMA_URL,
        )

    rag = LightRAG(
        working_dir=RAG_WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=RAG_LLM_MODEL,
        llm_model_max_async=1,
        embedding_func=embedding_func,
        enable_llm_cache=False,
        llm_model_kwargs={
            "host": OLLAMA_URL,
            "options": {
                "num_ctx": 8192,
                "temperature": 0.1,
            },
            "think": False,
        },
    )

    await rag.initialize_storages()
    logger.info("LightRAG graph database initialized.")


    print_section("Step 4: Ingesting into LightRAG")
    logger.info(f"Inserting document ({len(knowledge_doc)} chars) into knowledge graph...")

    await rag.ainsert(knowledge_doc)

    logger.info("Ingestion complete!")


    print_section("Step 5: Querying Knowledge Base")
    logger.info(f"Test Query: '{TEST_QUERY}' (Mode: hybrid)")

    try:
        answer = await rag.aquery(
            TEST_QUERY,
            param=QueryParam(
                mode="hybrid",
                top_k=20,
                enable_rerank=False,
            ),
        )
        print("\nLightRAG Svar:\n")
        print(answer)
    except Exception as e:
        logger.error(f"Query failed: {e}")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_pipeline())