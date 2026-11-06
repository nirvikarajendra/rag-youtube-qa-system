"""LCEL chains and the small helpers around them: context formatting, question
rewriting, query routing and map-reduce summarization."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from .prompts import ANSWER_PROMPT, CONDENSE_PROMPT, MAP_PROMPT, REDUCE_PROMPT
from .youtube import format_ts

_SUMMARY_RE = re.compile(
    r"\b(summari[sz]e|summary|tl;?dr|overview|main (points|topics|ideas|takeaways)|"
    r"key (points|topics|ideas|takeaways)|what (is|was) (this|the) video about|"
    r"what does (this|the) video cover|outline|recap)\b",
    re.IGNORECASE,
)


def route(question: str) -> str:
    """'summary' for whole-video questions that top-k retrieval can't answer, else 'qa'."""
    return "summary" if _SUMMARY_RE.search(question) else "qa"


def format_docs(docs: list[Document]) -> str:
    """Numbered excerpts with title and timestamp, so the model can cite [n]."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        blocks.append(f"[{i}] ({m.get('title', m['video_id'])} @ {format_ts(m['start'])})\n{doc.page_content}")
    return "\n\n".join(blocks)


def format_history(history: list[dict], max_turns: int) -> str:
    recent = history[-max_turns:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent)


def answer_chain(llm: BaseChatModel) -> Runnable:
    return ANSWER_PROMPT | llm | StrOutputParser()


def condense_question(llm: BaseChatModel, question: str, history: list[dict], max_turns: int) -> str:
    if not history:
        return question
    chain = CONDENSE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"history": format_history(history, max_turns), "question": question})
    return rewritten.strip() or question


def batch_chunks(chunks: list[Document], max_chars: int) -> list[list[Document]]:
    """Split chunks (already in video/time order) into consecutive batches per video."""
    batches: list[list[Document]] = []
    current: list[Document] = []
    size = 0
    for doc in chunks:
        new_video = current and doc.metadata["video_id"] != current[-1].metadata["video_id"]
        if current and (new_video or size + len(doc.page_content) > max_chars):
            batches.append(current)
            current, size = [], 0
        current.append(doc)
        size += len(doc.page_content)
    if current:
        batches.append(current)
    return batches


def dedupe_overlap(chunks: list[Document]) -> list[Document]:
    """Sort by video and time and drop exact duplicates (for map-reduce over all chunks)."""
    seen = set()
    ordered = []
    for doc in sorted(chunks, key=lambda d: (d.metadata["video_id"], d.metadata["start"])):
        if doc.metadata["chunk_id"] not in seen:
            seen.add(doc.metadata["chunk_id"])
            ordered.append(doc)
    return ordered


def map_summaries(llm: BaseChatModel, batches: list[list[Document]], max_concurrency: int = 4) -> list[str]:
    """Map step: one summary per batch, run in parallel."""
    chain = MAP_PROMPT | llm | StrOutputParser()
    inputs = [
        {
            "title": batch[0].metadata.get("title", batch[0].metadata["video_id"]),
            "text": "\n".join(f"[{format_ts(d.metadata['start'])}] {d.page_content}" for d in batch),
        }
        for batch in batches
    ]
    return chain.batch(inputs, config={"max_concurrency": max_concurrency})


def reduce_chain(llm: BaseChatModel) -> Runnable:
    return REDUCE_PROMPT | llm | StrOutputParser()
