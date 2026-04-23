#!/usr/bin/env python3
"""
LectoGraph — LightRAG query CLI.

Usage:
  python query.py "Hur öppnar man kontrollpanelen?"
  python query.py --mode global "Vilka kurser behandlar nätverkssäkerhet?"
  python query.py --mode naive "Windows 7"
  python query.py --top-k 30 "Vad är skillnaden mellan NTFS och FAT32?"
  python query.py --config other.yaml "..."

Query modes (LightRAG):
  hybrid  — combines local + global graph traversal  [default]
  local   — answers from nearby graph nodes
  global  — answers from high-level summaries
  naive   — plain vector similarity (fastest, least contextual)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query the LectoGraph LightRAG knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("query", nargs="+", help="The question to ask.")
    p.add_argument(
        "--config", default="./config.yaml", metavar="PATH",
        help="Path to config.yaml  (default: ./config.yaml)",
    )
    p.add_argument(
        "--mode", default="hybrid",
        choices=["hybrid", "local", "global", "naive"],
        help="LightRAG query mode  (default: hybrid)",
    )
    p.add_argument(
        "--top-k", type=int, default=20, metavar="N",
        help="Number of graph nodes to retrieve  (default: 20)",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Append a Swedish instruction to only answer from lesson material.",
    )
    p.add_argument(
        "--references", action="store_true",
        help="Include source references in the LLM response.",
    )
    return p.parse_args()


async def run_query(args: argparse.Namespace) -> int:
    from lectograph.config import Config

    try:
        cfg = Config.from_yaml(Path(args.config))
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not cfg.working_dir.exists():
        print(
            f"ERROR: Knowledge base not found at {cfg.working_dir.resolve()}\n"
            "Run  python ingest.py  first to build it.",
            file=sys.stderr,
        )
        return 1

    logging.basicConfig(
        level=logging.WARNING,       # suppress LightRAG chatter during queries
        format="%(levelname)s: %(message)s",
    )

    import numpy as np
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    @wrap_embedding_func_with_attrs(
        embedding_dim=cfg.rag_embedding_dim,
        max_token_size=8192,
        model_name=cfg.rag_embedding_model,
    )
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await ollama_embed.func(
            texts,
            embed_model=cfg.rag_embedding_model,
            host=cfg.ollama_url,
        )

    rag = LightRAG(
        working_dir=str(cfg.working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=cfg.rag_llm_model,
        llm_model_max_async=1,
        embedding_func=embedding_func,
        enable_llm_cache=False,
        llm_model_kwargs={
            "host": cfg.ollama_url,
            "options": {
                "num_ctx": cfg.rag_llm_num_ctx,
                "temperature": cfg.rag_llm_temperature,
            },
            "think": cfg.rag_llm_think,
        },
    )

    await rag.initialize_storages()

    query_text = " ".join(args.query)
    if args.strict:
        query_text += (
            "\n\nVIKTIGT: Basera ditt svar *endast* på informationen från lektionerna. "
            "Hitta inte på egen information eller allmän fakta om ämnet."
        )

    print(f"\nQuery  : {' '.join(args.query)}")
    print(f"Mode   : {args.mode}  |  top_k={args.top_k}")
    print("=" * 80)

    try:
        answer = await rag.aquery(
            query_text,
            param=QueryParam(
                mode=args.mode,
                top_k=args.top_k,
                enable_rerank=False,
                include_references=args.references,
            ),
        )
    except Exception as e:
        print(f"ERROR: Query failed — {e}", file=sys.stderr)
        return 1

    print(answer)
    print("=" * 80)
    return 0


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    args = parse_args()
    sys.exit(asyncio.run(run_query(args)))


if __name__ == "__main__":
    main()
