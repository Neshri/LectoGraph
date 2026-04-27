"""
Config loader for LectoGraph.
Reads config.yaml and exposes a typed Config dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    # ── Paths ─────────────────────────────────────────────────────────────────
    input_folder: Path = Path("./input")
    working_dir: Path = Path("./knowledge_db")
    docs_dir: Path = Path("./docs")
    logs_dir: Path = Path("./logs")

    # ── OpenSceneSense ────────────────────────────────────────────────────────
    whisper_model: str = "KBLab/kb-whisper-large"
    whisper_device: str = "cuda"
    # Vocabulary hint passed to Whisper as initial_prompt. Helps correct
    # accent-driven mishearings. Empty string = no prompt (Whisper default).
    whisper_initial_prompt: str = ""
    whisper_hotwords: str = ""
    frame_analysis_model: str = "glm-ocr"
    summary_model: str = "qwen3:32b"
    min_frames: int = 5
    max_frames: int = 45
    frames_per_minute: float = 3.0
    frame_threshold: float = 70.0
    request_timeout: float = 600.0
    request_retries: int = 1

    # ── LightRAG / Ollama ─────────────────────────────────────────────────────
    ollama_url: str = "http://127.0.0.1:11434"
    rag_llm_model: str = "qwen3:32b"
    rag_embedding_model: str = "qwen3-embedding:8b"
    rag_embedding_dim: int = 4096
    rag_llm_num_ctx: int = 8192
    rag_llm_temperature: float = 0.1
    rag_llm_think: bool = False
    rag_request_timeout: float = 600.0

    # ── Processing ────────────────────────────────────────────────────────────
    video_extensions: List[str] = field(
        default_factory=lambda: [".mp4", ".mkv", ".avi", ".mov", ".webm"]
    )

    # ── Prompts ───────────────────────────────────────────────────────────────
    # Prompts are defined in config.yaml to avoid duplicating them in code.
    frame_analysis_prompt: str = ""
    detailed_summary_prompt: str = ""
    brief_summary_prompt: str = ""

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def video_extensions_set(self) -> set:
        return {ext.lower() for ext in self.video_extensions}

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """
        Load config from a YAML file, falling back to defaults for any missing keys.

        Relative paths in the YAML are resolved relative to the config file's
        directory, NOT relative to the current working directory. This means
        the tool works correctly regardless of where you invoke it from.
        """
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                "Copy config.yaml to this location and edit it."
            )

        config_dir = path.parent

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Convert path-like values to Path objects, resolving relative paths
        # against the config file's directory (not CWD).
        path_keys = ("input_folder", "working_dir", "docs_dir", "logs_dir")
        for key in path_keys:
            if key in data:
                p = Path(data[key])
                data[key] = (config_dir / p).resolve() if not p.is_absolute() else p

        # Only pass recognised fields so unknown YAML keys don't crash the init
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered)

    def summary(self) -> str:
        """Human-readable config summary for logging."""
        return (
            f"  input_folder        : {self.input_folder}\n"
            f"  working_dir         : {self.working_dir}\n"
            f"  docs_dir            : {self.docs_dir}\n"
            f"  logs_dir            : {self.logs_dir}\n"
            f"  whisper_model       : {self.whisper_model} ({self.whisper_device})\n"
            f"  frame_analysis_model: {self.frame_analysis_model}\n"
            f"  summary_model       : {self.summary_model}\n"
            f"  rag_llm_model       : {self.rag_llm_model}\n"
            f"  rag_embedding_model : {self.rag_embedding_model} (dim={self.rag_embedding_dim})\n"
            f"  ollama_url          : {self.ollama_url}\n"
        )
