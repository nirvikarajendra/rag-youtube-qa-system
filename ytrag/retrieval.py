"""Retrieval strategies.

similarity : top-k FAISS nearest neighbours (the baseline)
mmr        : Maximal Marginal Relevance over fetch_k candidates, which drops near-duplicates
hybrid     : BM25 keyword search + FAISS, merged with Reciprocal Rank Fusion
rerank     : optional LLM pass that reorders fetch_k candidates and keeps the best k
"""

from __future__ import annotations

import json
import logging
import re

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from .config import RAGConfig
from .prompts import RERANK_PROMPT

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], top_n: int, rrf_k: int = 60
) -> list[Document]:
    """score(d) = sum over lists of 1 / (rrf_k + rank). Rank-based, so BM25 and cosine
    scores never need to be put on the same scale."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.metadata["chunk_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered[:top_n]]


class Retriever:
    def __init__(
        self,
        store: FAISS,
        chunks: list[Document],
        config: RAGConfig,
        llm: BaseChatModel | None = None,
    ):
        self.store = store
        self.chunks = chunks
        self.config = config
        self.llm = llm
        self._bm25: BM25Retriever | None = None
        if config.retrieval == "hybrid":
            self._bm25 = BM25Retriever.from_documents(chunks, preprocess_func=tokenize)
        if config.rerank and llm is None:
            raise ValueError("rerank=True needs an LLM")

    def vector_search(self, query: str, k: int) -> list[Document]:
        if self.config.retrieval == "mmr":
            return self.store.max_marginal_relevance_search(
                query, k=k, fetch_k=max(self.config.fetch_k, k)
            )
        return self.store.similarity_search(query, k=k)

    def bm25_search(self, query: str, k: int) -> list[Document]:
        assert self._bm25 is not None
        self._bm25.k = k
        return self._bm25.invoke(query)

    def retrieve(self, query: str) -> list[Document]:
        cfg = self.config
        # With reranking, gather a wider pool first and let the reranker pick the final k.
        n = cfg.fetch_k if cfg.rerank else cfg.k

        if cfg.retrieval == "hybrid":
            pool = reciprocal_rank_fusion(
                [self.vector_search(query, cfg.fetch_k), self.bm25_search(query, cfg.fetch_k)],
                top_n=n,
            )
        else:
            pool = self.vector_search(query, n)

        if cfg.rerank:
            return self.rerank(query, pool, cfg.k)
        return pool

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """One LLM call: the model returns passage numbers ordered by relevance."""
        if len(docs) <= 1:
            return docs[:top_n]
        passages = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs))
        chain = RERANK_PROMPT | self.llm | StrOutputParser()
        raw = chain.invoke({"question": query, "passages": passages, "top_n": top_n})
        try:
            match = re.search(r"\[[\d,\s]*\]", raw)
            order = json.loads(match.group(0)) if match else []
        except json.JSONDecodeError:
            order = []
        picked, seen = [], set()
        for i in order:
            if isinstance(i, int) and 0 <= i < len(docs) and i not in seen:
                picked.append(docs[i])
                seen.add(i)
        if not picked:
            log.warning("Reranker output unparseable; keeping retrieval order")
        # Fill from the original order if the model returned fewer than top_n.
        for i, d in enumerate(docs):
            if len(picked) >= top_n:
                break
            if i not in seen:
                picked.append(d)
        return picked[:top_n]
