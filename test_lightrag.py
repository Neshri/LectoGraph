#!/usr/bin/env python3
"""
Standalone external test for LightRAG + Ollama integration.

This script bypasses the full LessonTranscriber pipeline and tests
LightRAG directly so we can validate:
  1. LightRAG initializes correctly with our Ollama LLM + embedding setup
  2. Documents can be inserted (ainsert)
  3. Queries return sensible results in all supported modes
  4. The JSON output format we expect actually works

Usage:
    python test_lightrag.py

Prerequisites:
    - Ollama is running at http://127.0.0.1:11434
    - The LLM model (gpt-oss:20b) is pulled in Ollama
    - The embedding model (nomic-embed-text) is pulled in Ollama
    - lightrag-hku is installed  (pip install lightrag-hku)
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import numpy as np
from pathlib import Path

# ─── Configuration (mirrors config.json) ────────────────────────────────────
OLLAMA_URL        = "http://127.0.0.1:11434"
LLM_MODEL         = "qwen3:32b"
EMBEDDING_MODEL   = "nomic-embed-text"
EMBEDDING_DIM     = 768          # nomic-embed-text output dimension
WORKING_DIR       = "./test_lightrag_storage"

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_lightrag")

# ─── Sample transcript (full Whisper transcript) ─────────────────────────────
with open(Path(__file__).parent / "test_transcript.txt", "r", encoding="utf-8") as f:
    SAMPLE_TRANSCRIPT = f.read()

# NOTE: The JSON schema literal { ... } in the query string confuses LightRAG's
# keyword extractor (hybrid/global/local modes), producing empty keywords → no
# context → "no-context" refusal.  Step 7 tests JSON *output formatting*, not
# query modes (those are covered by Step 6), so we use naive mode (vector-only,
# no keyword extraction) and keep the query plain.
QUERY_JSON = (
    "Vad handlade lektionen om? "
    "Skapa en tekniskt korrekt och kortfattad sammanfattning på svenska. "
    "Svara ENDAST med ett JSON-objekt i exakt detta format: "
    '{{"subject": "Ämnesrad", "summary": "- Punkt 1\\n- Punkt 2"}}'
)

QUERY_PLAIN = "Vad handlade lektionen om? Besvara på svenska."


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_section(title: str):
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def cleanup_working_dir():
    if Path(WORKING_DIR).exists():
        shutil.rmtree(WORKING_DIR)
        logger.info(f"Cleaned up working dir: {WORKING_DIR}")
    Path(WORKING_DIR).mkdir(parents=True, exist_ok=True)


# ─── Main async test ─────────────────────────────────────────────────────────

async def run_test():
    # Late import so the script still gives a helpful error if not installed
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.ollama import ollama_model_complete, ollama_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        logger.error("Make sure lightrag-hku is installed:  pip install lightrag-hku")
        sys.exit(1)

    # ── Step 1: clean slate ───────────────────────────────────────────────
    print_section("Step 1: Preparing working directory")
    cleanup_working_dir()

    # ── Step 2: build embedding function ─────────────────────────────────
    print_section("Step 2: Building embedding function")

    @wrap_embedding_func_with_attrs(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await ollama_embed.func(
            texts,
            embed_model=EMBEDDING_MODEL,
            host=OLLAMA_URL,
        )

    logger.info(f"Embedding model : {EMBEDDING_MODEL}")
    logger.info(f"Embedding dim   : {EMBEDDING_DIM}")

    # ── Step 3: create LightRAG instance ─────────────────────────────────
    print_section("Step 3: Creating LightRAG instance")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_max_async=1,  # Prevent 4x parallel slowdown hitting the 360s timeout
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
    logger.info(f"LLM model       : {LLM_MODEL}")
    logger.info(f"Ollama URL      : {OLLAMA_URL}")
    logger.info(f"Working dir     : {WORKING_DIR}")

    # ── Step 4: initialize storages ───────────────────────────────────────
    print_section("Step 4: Initializing LightRAG storages")
    await rag.initialize_storages()
    logger.info("Storages initialized OK")

    # ── Step 5: insert document ───────────────────────────────────────────
    print_section("Step 5: Inserting sample transcript")
    logger.info(f"Transcript length: {len(SAMPLE_TRANSCRIPT)} chars")
    await rag.ainsert(SAMPLE_TRANSCRIPT)
    logger.info("Insert complete")

    # ── Step 6: query in different modes ─────────────────────────────────
    results = {}

    for mode in ["naive", "local", "global", "hybrid"]:
        print_section(f"Step 6-{mode}: Query mode={mode}")
        try:
            answer = await rag.aquery(
                QUERY_PLAIN,
                param=QueryParam(
                    mode=mode,
                    top_k=10,
                    enable_rerank=False,
                ),
            )
            logger.info(f"[{mode}] Response:\n{answer}")
            results[mode] ={"ok": True, "response": answer}
        except Exception as e:
            logger.error(f"[{mode}] FAILED: {e}")
            results[mode] = {"ok": False, "error": str(e)}

    # ── Step 7: JSON query (mirrors production usage) ─────────────────────
    # Use naive mode: vector-only search avoids the keyword-extractor bug that
    # fires when the query contains JSON literals.  The JSON *formatting* is
    # what Step 7 validates; query-mode coverage is handled by Step 6.
    print_section("Step 7: JSON-format query (naive mode, production-style)")
    try:
        json_answer = await rag.aquery(
            QUERY_JSON,
            param=QueryParam(
                mode="naive",
                top_k=20,
                response_type="Single JSON object",
                enable_rerank=False,
            ),
        )
        logger.info(f"Raw JSON response:\n{json_answer}")

        # Try to parse it; fall back to extracting a JSON block from prose/fences
        import re
        json_candidate = json_answer.strip()
        if not json_candidate.startswith("{"):
            m = re.search(r"```(?:json)?\s*({.*?})\s*```", json_candidate, re.DOTALL)
            if not m:
                m = re.search(r"({.*})", json_candidate, re.DOTALL)
            json_candidate = m.group(1) if m else json_candidate

        try:
            parsed = json.loads(json_candidate)
            subject = parsed.get("subject")
            summary = parsed.get("summary", "")
            if not subject or not summary:
                logger.warning(
                    f"⚠️  JSON parsed but schema is wrong — expected 'subject' and "
                    f"'summary' keys, got: {list(parsed.keys())}"
                )
                results["json_query"] = {"ok": False, "raw": json_answer, "parse_error": "wrong schema"}
            else:
                logger.info(f"✅  JSON parsed OK!")
                logger.info(f"    subject : {subject}")
                logger.info(f"    summary : {summary[:200]}")
                results["json_query"] = {"ok": True, "parsed": parsed}
        except json.JSONDecodeError as je:
            logger.warning(f"⚠️  JSON parse failed ({je}). Raw output saved.")
            results["json_query"] = {"ok": False, "raw": json_answer, "parse_error": str(je)}
    except Exception as e:
        logger.error(f"JSON query FAILED: {e}")
        results["json_query"] = {"ok": False, "error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────
    print_section("Test Summary")
    all_ok = all(v["ok"] for v in results.values())
    for step, res in results.items():
        status = "✅ OK" if res["ok"] else "❌ FAIL"
        logger.info(f"  {step:12s} {status}")

    if all_ok:
        logger.info("\n✅  All steps passed!")
    else:
        logger.warning("\n⚠️  Some steps failed — see output above for details.")

    return results


if __name__ == "__main__":
    asyncio.run(run_test())
