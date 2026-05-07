from __future__ import annotations

import logging
import numpy as np

from .config import Config


# ─── Analyzer factory ─────────────────────────────────────────────────────────

def build_analyzer(cfg: Config, logger: logging.Logger):
    """
    Construct and return a fully-initialised OllamaVideoAnalyzer.
    Loading Whisper is the expensive part — do this once before the loop.
    """
    from openscenesense_ollama.models import AnalysisPrompts
    from lectograph.transcriber import FasterWhisperAdapter
    from openscenesense_ollama.analyzer import OllamaVideoAnalyzer
    from openscenesense_ollama.frame_selectors import DynamicFrameSelector

    logger.info(f"Loading Whisper model: {cfg.whisper_model} (device={cfg.whisper_device})")

    custom_prompts = AnalysisPrompts(
        frame_analysis=cfg.frame_analysis_prompt,
        detailed_summary=cfg.detailed_summary_prompt,
        brief_summary=cfg.brief_summary_prompt,
    )

    transcriber = FasterWhisperAdapter(
        model_name=cfg.whisper_model,
        device=cfg.whisper_device,
        initial_prompt=cfg.whisper_initial_prompt or None,
        hotwords=cfg.whisper_hotwords or None,
    )

    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=cfg.frame_analysis_model,
        summary_model=cfg.summary_model,
        min_frames=cfg.min_frames,
        max_frames=cfg.max_frames,
        frames_per_minute=cfg.frames_per_minute,
        frame_selector=DynamicFrameSelector(threshold=cfg.frame_threshold),
        audio_transcriber=transcriber,
        prompts=custom_prompts,
        request_timeout=cfg.request_timeout,
        request_retries=cfg.request_retries,
        log_level=logging.INFO,
    )

    logger.info("Analyzer ready.")
    return analyzer


# ─── LightRAG factory ─────────────────────────────────────────────────────────

async def build_rag(cfg: Config, logger: logging.Logger):
    """Construct and initialise a LightRAG instance."""
    from lightrag import LightRAG
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    logger.info(
        f"Initialising LightRAG at {cfg.working_dir}  "
        f"(llm={cfg.rag_llm_model}, embed={cfg.rag_embedding_model})"
    )

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
        default_embedding_timeout=int(cfg.rag_request_timeout),
        default_llm_timeout=int(cfg.rag_request_timeout),
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
    logger.info("LightRAG initialised.")
    return rag
