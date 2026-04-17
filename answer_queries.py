#!/usr/bin/env python3
import sys
import asyncio
import logging
import re
import yaml
import os
import numpy as np

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Answer_Queries")

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_questions(queries_file="queries.txt"):
    if not os.path.exists(queries_file):
        logger.error(f"Queries file {queries_file} not found. Please run generate_query.py first.")
        sys.exit(1)
        
    questions = []
    # Match patterns like: "    - Fråga 1: Vilka undervisningsmetoder..."
    pattern = re.compile(r"Fråga\s+\d+:\s*(.+)")
    
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                questions.append(match.group(1).strip())
                
    return questions

async def process_queries():
    config = load_config()
    
    OLLAMA_URL = config.get("ollama_url", "http://127.0.0.1:11434")
    RAG_LLM_MODEL = config.get("rag_llm_model", "qwen3:32b")
    RAG_EMBEDDING_MODEL = config.get("rag_embedding_model", "qwen3-embedding:8b")
    RAG_EMBEDDING_DIM = config.get("rag_embedding_dim", 4096)
    RAG_WORKING_DIR = config.get("working_dir", "./knowledge_db")
    RAG_LLM_NUM_CTX = config.get("rag_llm_num_ctx", 8192)
    RAG_LLM_TEMPERATURE = config.get("rag_llm_temperature", 0.1)

    logger.info(f"Loading LightRAG database from: {RAG_WORKING_DIR}")
    
    @wrap_embedding_func_with_attrs(
        embedding_dim=RAG_EMBEDDING_DIM,
        max_token_size=RAG_LLM_NUM_CTX,
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
                "num_ctx": RAG_LLM_NUM_CTX,
                "temperature": RAG_LLM_TEMPERATURE, 
            },
            "think": False,
        },
    )
    
    await rag.initialize_storages()
    
    questions = extract_questions()
    logger.info(f"Extracted {len(questions)} questions from queries.txt")
    
    if not questions:
        logger.warning("No questions found. Exiting.")
        return

    output_file = "answers.txt"
    # Create or overwrite the answers file initially with a header
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# LightRAG Query Answers\n\n")

    for i, question in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}: {question}")
        
        try:
            answer = await rag.aquery(
                question,
                param=QueryParam(
                    mode="hybrid",
                    top_k=20,
                    enable_rerank=False,
                ),
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            answer = f"[ERROR: Query failed - {e}]"
            
        # Append answer progressively so no data is lost if the script stops
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"### Q{i}: {question}\n\n")
            f.write(f"**Svar:**\n{answer}\n\n")
            f.write("---\n\n")
            
    logger.info(f"Finished processing all {len(questions)} questions. Results saved to {output_file}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(process_queries())
