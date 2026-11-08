"""Central configuration.

`Settings` holds environment-level values (API key, model names, paths).
`RAGConfig` holds the pipeline knobs that the evaluation varies between runs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


@dataclass(frozen=True)
class Settings:
    chat_model: str = field(default_factory=lambda: _env("YTRAG_CHAT_MODEL", "gemini-3.5-flash-lite"))
    judge_model: str = field(default_factory=lambda: _env("YTRAG_JUDGE_MODEL", "gemini-3.5-flash-lite"))
    embedding_model: str = field(
        default_factory=lambda: _env("YTRAG_EMBEDDING_MODEL", "models/gemini-embedding-001")
    )
    temperature: float = field(default_factory=lambda: float(_env("YTRAG_TEMPERATURE", "0")))
    data_dir: Path = field(
        default_factory=lambda: Path(_env("YTRAG_DATA_DIR", str(PROJECT_ROOT / "data")))
    )
    transcript_languages: tuple[str, ...] = field(
        default_factory=lambda: tuple(_env("YTRAG_LANGUAGES", "en,en-US,en-GB").split(","))
    )
    max_video_minutes: int = field(default_factory=lambda: int(_env("YTRAG_MAX_VIDEO_MINUTES", "180")))
    history_turns: int = field(default_factory=lambda: int(_env("YTRAG_HISTORY_TURNS", "6")))

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def indexes_dir(self) -> Path:
        # Indexes are tied to the embedding model, so a model change never mixes vectors.
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.embedding_model)
        return self.data_dir / "indexes" / slug

    def require_api_key(self) -> None:
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key "
                "from https://aistudio.google.com/apikey"
            )


CHUNKERS = ("char", "time")
RETRIEVAL_MODES = ("similarity", "mmr", "hybrid")


@dataclass(frozen=True)
class RAGConfig:
    """Pipeline knobs. Defaults are the app's best setup; eval/configs.yaml overrides them."""

    chunker: str = "time"  # "char" = fixed-size baseline, "time" = timestamp windows
    chunk_size: int = 1000  # char chunker: chunk size; time chunker: max chars per chunk
    chunk_overlap: int = 100  # char chunker only
    window_seconds: int = 60  # time chunker only
    overlap_segments: int = 1  # time chunker only: segments repeated between windows
    retrieval: str = "hybrid"  # similarity | mmr | hybrid (BM25 + FAISS with RRF)
    k: int = 4  # chunks given to the LLM
    fetch_k: int = 20  # candidates fetched before MMR / fusion / rerank
    rerank: bool = False  # LLM reranks fetch_k candidates down to k
    router: bool = True  # send summary-style questions to map-reduce
    map_batch_chars: int = 8000  # map-reduce: transcript chars per map call

    def __post_init__(self) -> None:
        if self.chunker not in CHUNKERS:
            raise ValueError(f"chunker must be one of {CHUNKERS}, got {self.chunker!r}")
        if self.retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval must be one of {RETRIEVAL_MODES}, got {self.retrieval!r}")
        if self.k < 1 or self.fetch_k < self.k:
            raise ValueError("need 1 <= k <= fetch_k")

    @property
    def chunk_signature(self) -> str:
        """Folder name for an index; different chunking means a different index."""
        if self.chunker == "char":
            return f"char-{self.chunk_size}-{self.chunk_overlap}"
        return f"time-{self.window_seconds}s-{self.chunk_size}-{self.overlap_segments}"

    def with_overrides(self, **overrides) -> "RAGConfig":
        return replace(self, **overrides)
