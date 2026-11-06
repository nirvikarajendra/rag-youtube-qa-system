"""Build, persist and load one FAISS index per (video, chunking config).

Layout:  data/indexes/<embedding-model>/<video_id>/<chunk-signature>/
           index.faiss, index.pkl   (FAISS.save_local)
           chunks.json              (all chunks, for BM25 and map-reduce summaries)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .chunking import chunk_transcript
from .config import RAGConfig, Settings
from .youtube import Transcript, TranscriptError, load_transcript

log = logging.getLogger(__name__)


@dataclass
class VideoIndex:
    video_id: str
    title: str
    store: FAISS
    chunks: list[Document]


def index_path(settings: Settings, video_id: str, config: RAGConfig) -> Path:
    return settings.indexes_dir / video_id / config.chunk_signature


def is_indexed(settings: Settings, video_id: str, config: RAGConfig) -> bool:
    path = index_path(settings, video_id, config)
    return (path / "index.faiss").exists() and (path / "chunks.json").exists()


def _check_length(transcript: Transcript, settings: Settings) -> None:
    minutes = transcript.duration / 60
    if minutes > settings.max_video_minutes:
        raise TranscriptError(
            f"Video is {minutes:.0f} min long; the limit is {settings.max_video_minutes} min "
            "(YTRAG_MAX_VIDEO_MINUTES)."
        )


def build_or_load_index(
    video_id: str,
    settings: Settings,
    config: RAGConfig,
    embeddings: Embeddings,
    force: bool = False,
) -> VideoIndex:
    """Load the saved index, or fetch → chunk → embed → save it. Never re-embeds unless forced."""
    path = index_path(settings, video_id, config)
    if is_indexed(settings, video_id, config) and not force:
        store = FAISS.load_local(
            # index.pkl is a pickle; safe here because this app wrote it, never load untrusted ones.
            str(path), embeddings, allow_dangerous_deserialization=True
        )
        data = json.loads((path / "chunks.json").read_text())
        chunks = [Document(page_content=c["text"], metadata=c["metadata"]) for c in data["chunks"]]
        return VideoIndex(video_id, data["title"], store, chunks)

    transcript = load_transcript(video_id, settings.transcripts_dir, settings.transcript_languages)
    _check_length(transcript, settings)
    chunks = chunk_transcript(transcript, config)
    log.info("Embedding %d chunks for %s (%s)", len(chunks), video_id, config.chunk_signature)
    store = FAISS.from_documents(chunks, embeddings, ids=[c.metadata["chunk_id"] for c in chunks])

    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    (path / "chunks.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "title": transcript.title,
                "config": config.chunk_signature,
                "chunks": [{"text": c.page_content, "metadata": c.metadata} for c in chunks],
            },
            ensure_ascii=False,
        )
    )
    return VideoIndex(video_id, transcript.title, store, chunks)
