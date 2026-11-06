"""Transcript chunkers. Every chunk keeps video_id, start and end so answers can cite timestamps.

- `char_chunks`: fixed-size chunks with RecursiveCharacterTextSplitter, kept as the eval
  baseline. Timestamps are recovered by mapping each chunk's character offset back to the
  transcript segment it starts in.
- `time_window_chunks`: groups whole caption segments into ~N-second windows, so chunks
  never cut a sentence fragment mid-segment and timestamps are exact.
"""

from __future__ import annotations

from bisect import bisect_right

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAGConfig
from .youtube import Segment, Transcript, video_url


def _make_doc(transcript: Transcript, index: int, text: str, start: float, end: float) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": f"{transcript.video_id}:{index}",
            "video_id": transcript.video_id,
            "title": transcript.title,
            "start": round(start, 2),
            "end": round(end, 2),
            "url": video_url(transcript.video_id, start),
        },
    )


def char_chunks(transcript: Transcript, chunk_size: int = 500, chunk_overlap: int = 20) -> list[Document]:
    segments = transcript.segments
    # Join with single spaces and remember where each segment starts in the joined text.
    offsets: list[int] = []
    pieces: list[str] = []
    pos = 0
    for seg in segments:
        offsets.append(pos)
        pieces.append(seg.text)
        pos += len(seg.text) + 1
    full_text = " ".join(pieces)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    docs = []
    for i, piece in enumerate(splitter.create_documents([full_text])):
        start_char = piece.metadata["start_index"]
        end_char = start_char + len(piece.page_content) - 1
        first = segments[max(bisect_right(offsets, start_char) - 1, 0)]
        last = segments[max(bisect_right(offsets, end_char) - 1, 0)]
        docs.append(_make_doc(transcript, i, piece.page_content, first.start, last.end))
    return docs


def time_window_chunks(
    transcript: Transcript, window_seconds: int = 60, max_chars: int = 1000, overlap_segments: int = 1
) -> list[Document]:
    docs: list[Document] = []
    current: list[Segment] = []
    current_chars = 0
    fresh = 0  # segments added since the last flush (excludes carried-over overlap)

    def flush() -> None:
        text = " ".join(s.text for s in current)
        docs.append(_make_doc(transcript, len(docs), text, current[0].start, current[-1].end))

    for seg in transcript.segments:
        too_long = current and seg.start - current[0].start >= window_seconds
        too_big = current and current_chars + len(seg.text) + 1 > max_chars
        if too_long or too_big:
            if fresh:
                flush()
                current = current[-overlap_segments:] if overlap_segments else []
            else:
                current = []  # only overlap left; don't emit a duplicate chunk
            current_chars = sum(len(s.text) + 1 for s in current)
            fresh = 0
        current.append(seg)
        current_chars += len(seg.text) + 1
        fresh += 1
    if fresh:
        flush()
    return docs


def chunk_transcript(transcript: Transcript, config: RAGConfig) -> list[Document]:
    if config.chunker == "char":
        return char_chunks(transcript, config.chunk_size, config.chunk_overlap)
    return time_window_chunks(
        transcript, config.window_seconds, config.chunk_size, config.overlap_segments
    )
