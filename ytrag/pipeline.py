"""RAGPipeline: the one object shared by the CLI, Streamlit UI and the RAGAS eval,
so the scores always measure the code the app actually runs.

    pipeline = RAGPipeline()
    pipeline.ingest("https://youtu.be/B5j3uNBH8X4")
    result = pipeline.ask("What is a producer?", ["B5j3uNBH8X4"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from . import chains
from .config import RAGConfig, Settings
from .index import VideoIndex, build_or_load_index, index_path
from .retrieval import Retriever
from .youtube import extract_video_id, format_ts, list_cached_transcripts, video_url


@dataclass
class Source:
    n: int
    video_id: str
    title: str
    start: float
    end: float
    timestamp: str
    url: str
    text: str


@dataclass
class Answer:
    answer: str
    question: str  # the standalone question actually used for retrieval
    route: str  # "qa" or "summary"
    sources: list[Source] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # what the LLM saw; used by RAGAS


@dataclass
class PreparedQuery:
    """Everything except the final generation step, so ask() and stream() share one path."""

    question: str
    route: str
    sources: list[Source]
    contexts: list[str]
    chain: object
    inputs: dict

    def stream(self) -> Iterator[str]:
        yield from self.chain.stream(self.inputs)

    def invoke(self) -> str:
        return self.chain.invoke(self.inputs)


def _source(n: int, doc: Document) -> Source:
    m = doc.metadata
    return Source(
        n=n,
        video_id=m["video_id"],
        title=m.get("title", m["video_id"]),
        start=m["start"],
        end=m["end"],
        timestamp=format_ts(m["start"]),
        url=video_url(m["video_id"], m["start"]),
        text=doc.page_content,
    )


def default_llm(settings: Settings, model: str | None = None) -> BaseChatModel:
    settings.require_api_key()
    return ChatGoogleGenerativeAI(
        model=model or settings.chat_model, temperature=settings.temperature, max_retries=3
    )


def default_embeddings(settings: Settings) -> Embeddings:
    settings.require_api_key()
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig | None = None,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ):
        self.settings = settings or Settings()
        self.config = config or RAGConfig()
        # Injectable so tests can run offline with fake models.
        self.llm = llm or default_llm(self.settings)
        self.embeddings = embeddings or default_embeddings(self.settings)
        self._indexes: dict[str, VideoIndex] = {}
        self._retrievers: dict[tuple[str, ...], Retriever] = {}

    # ---------- ingestion ----------

    def ingest(self, url_or_id: str, force: bool = False) -> dict:
        video_id = extract_video_id(url_or_id)
        index = build_or_load_index(video_id, self.settings, self.config, self.embeddings, force=force)
        self._indexes[video_id] = index
        self._retrievers = {k: v for k, v in self._retrievers.items() if video_id not in k}
        return {"video_id": video_id, "title": index.title, "chunks": len(index.chunks)}

    def list_videos(self) -> list[dict]:
        return list_cached_transcripts(self.settings.transcripts_dir)

    def _index(self, video_id: str) -> VideoIndex:
        if video_id not in self._indexes:
            self._indexes[video_id] = build_or_load_index(
                video_id, self.settings, self.config, self.embeddings
            )
        return self._indexes[video_id]

    def retriever(self, video_ids: list[str]) -> Retriever:
        key = tuple(sorted(set(video_ids)))
        if not key:
            raise ValueError("Select at least one video.")
        if key not in self._retrievers:
            indexes = [self._index(v) for v in key]
            store = indexes[0].store
            if len(indexes) > 1:
                # merge_from mutates its target, so merge into a fresh copy loaded from disk.
                store = FAISS.load_local(
                    str(index_path(self.settings, key[0], self.config)),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                for idx in indexes[1:]:
                    store.merge_from(idx.store)
            chunks = [c for idx in indexes for c in idx.chunks]
            self._retrievers[key] = Retriever(store, chunks, self.config, llm=self.llm)
        return self._retrievers[key]

    # ---------- question answering ----------

    def prepare(self, question: str, video_ids: list[str], history: list[dict] | None = None) -> PreparedQuery:
        """Rewrite the question, pick a route, and gather everything the final LLM call needs."""
        standalone = chains.condense_question(
            self.llm, question, history or [], self.settings.history_turns
        )
        retriever = self.retriever(video_ids)
        if self.config.router and chains.route(standalone) == "summary":
            return self._prepare_summary(standalone, retriever)
        return self._prepare_qa(standalone, retriever)

    def _prepare_qa(self, question: str, retriever: Retriever) -> PreparedQuery:
        """Normal question: retrieve the top-k chunks and answer from them."""
        docs = retriever.retrieve(question)
        return PreparedQuery(
            question=question,
            route="qa",
            sources=[_source(i, d) for i, d in enumerate(docs, start=1)],
            contexts=[d.page_content for d in docs],
            chain=chains.answer_chain(self.llm),
            inputs={"context": chains.format_docs(docs), "question": question},
        )

    def _prepare_summary(self, question: str, retriever: Retriever) -> PreparedQuery:
        """Whole-video question: summarize every batch of chunks (map), then combine (reduce)."""
        chunks = chains.dedupe_overlap(retriever.chunks)
        batches = chains.batch_chunks(chunks, self.config.map_batch_chars)
        notes = chains.map_summaries(self.llm, batches)
        sections = []
        for i, (batch, note) in enumerate(zip(batches, notes), start=1):
            first = batch[0].metadata
            sections.append(f"Section {i} ({first.get('title')} from {format_ts(first['start'])}):\n{note}")
        return PreparedQuery(
            question=question,
            route="summary",
            sources=[_source(i, batch[0]) for i, batch in enumerate(batches, start=1)],
            contexts=notes,
            chain=chains.reduce_chain(self.llm),
            inputs={"notes": "\n\n".join(sections), "question": question},
        )

    def ask(self, question: str, video_ids: list[str], history: list[dict] | None = None) -> Answer:
        prepared = self.prepare(question, video_ids, history)
        return Answer(
            answer=prepared.invoke(),
            question=prepared.question,
            route=prepared.route,
            sources=prepared.sources,
            contexts=prepared.contexts,
        )
