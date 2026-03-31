#!/usr/bin/env python3
"""
Basic CLI test script for querying the LightRAG course database.
Usage: python test_query_rag.py "Your question here"
"""

import sys
import asyncio
import logging
import numpy as np

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────

OLLAMA_URL = "http://127.0.0.1:11434"
RAG_LLM_MODEL = "qwen3:32b"
RAG_EMBEDDING_MODEL = "qwen3-embedding:8b"
RAG_EMBEDDING_DIM = 4096
RAG_WORKING_DIR = "./test_course_knowledge_db"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_Query")

async def run_query(query_text: str):
    logger.info(f"Loading LightRAG database from: {RAG_WORKING_DIR}")
    
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
    
    logger.info(f"Submitting query: '{query_text}'")
    
    # Prompting for strict factual retrieval
    strict_query = (
        f"{query_text}\n\n"
        "VIKTIGT: Basera ditt svar *endast* på informationen från lektionen. "
        "Hitta inte på egen information eller allmän fakta om ämnet."
    )
    
    try:
        answer = await rag.aquery(
            strict_query,
            param=QueryParam(
                mode="hybrid",
                top_k=20,
                enable_rerank=False,
            ),
        )
        
        print("\n" + "="*80)
        print("SVAR:")
        print("="*80 + "\n")
        print(answer)
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    # Get query from command line arguments or use default
    if len(sys.argv) > 1:
        query_input = " ".join(sys.argv[1:])
    else:
        query_input = "Hur öppnar man kontrollpanelen och vilka inställningar syns på skärmen?"
        
    asyncio.run(run_query(query_input))
