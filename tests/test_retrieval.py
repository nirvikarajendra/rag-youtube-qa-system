import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ytrag.chains import route
from ytrag.chunking import time_window_chunks
from ytrag.config import RAGConfig
from ytrag.retrieval import Retriever, reciprocal_rank_fusion, tokenize


def doc(cid: str) -> Document:
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([[doc("a"), doc("b"), doc("c")], [doc("b"), doc("d"), doc("a")]], top_n=4)
    assert [d.metadata["chunk_id"] for d in fused][:2] == ["b", "a"]
    assert len(fused) == 4


def test_tokenize():
    assert tokenize("Kafka's ZooKeeper, KIP-500!") == ["kafka", "s", "zookeeper", "kip", "500"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Summarize the video", "summary"),
        ("Can u summarise the video?", "summary"),
        ("What are the main topics?", "summary"),
        ("give me a tl;dr", "summary"),
        ("What is a producer?", "qa"),
        ("How does replication work?", "qa"),
    ],
)
def test_router(question, expected):
    assert route(question) == expected


@pytest.fixture
def chunks(transcript):
    return time_window_chunks(transcript, window_seconds=30, max_chars=1000)


@pytest.mark.parametrize("mode", ["similarity", "mmr", "hybrid"])
def test_retriever_modes_return_k(chunks, embeddings, mode):
    store = FAISS.from_documents(chunks, embeddings)
    r = Retriever(store, chunks, RAGConfig(retrieval=mode, k=3, fetch_k=8))
    docs = r.retrieve("What is ZooKeeper?")
    assert len(docs) == 3
    assert len({d.metadata["chunk_id"] for d in docs}) == 3


def test_bm25_finds_keyword(chunks, embeddings):
    store = FAISS.from_documents(chunks, embeddings)
    r = Retriever(store, chunks, RAGConfig(retrieval="hybrid", k=3))
    top = r.bm25_search("zookeeper leader election", k=1)[0]
    assert "ZooKeeper" in top.page_content


def test_rerank_uses_llm_order_and_fills_gaps(chunks, embeddings, fake_llm):
    store = FAISS.from_documents(chunks, embeddings)
    cfg = RAGConfig(retrieval="similarity", k=3, fetch_k=6, rerank=True)
    r = Retriever(store, chunks, cfg, llm=fake_llm(["Ranking: [4, 4, 99, 1]"]))
    pool = r.vector_search("q", 6)
    out = r.rerank("q", pool, 3)
    assert out[0] is pool[4] and out[1] is pool[1]  # duplicates and out-of-range ignored
    assert out[2] is pool[0]  # filled from original order
