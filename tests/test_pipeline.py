from ytrag.chains import batch_chunks, dedupe_overlap
from ytrag.config import RAGConfig
from ytrag.index import index_path, is_indexed
from ytrag.pipeline import RAGPipeline

from .conftest import VIDEO_ID


def make_pipeline(settings, embeddings, llm, **overrides):
    return RAGPipeline(RAGConfig().with_overrides(**overrides), settings, llm=llm, embeddings=embeddings)


def test_ingest_persists_and_reloads(settings, embeddings, fake_llm):
    p = make_pipeline(settings, embeddings, fake_llm(["x"]))
    info = p.ingest(f"https://youtu.be/{VIDEO_ID}")
    assert info["video_id"] == VIDEO_ID and info["chunks"] > 0
    assert is_indexed(settings, VIDEO_ID, p.config)
    assert (index_path(settings, VIDEO_ID, p.config) / "chunks.json").exists()

    # A new pipeline loads from disk instead of rebuilding.
    p2 = make_pipeline(settings, embeddings, fake_llm(["x"]))
    assert p2.ingest(VIDEO_ID)["chunks"] == info["chunks"]
    assert p2.list_videos()[0]["title"] == "Kafka Fundamentals"


def test_different_chunking_gets_its_own_index(settings, embeddings, fake_llm):
    a = make_pipeline(settings, embeddings, fake_llm(["x"]), chunker="char", chunk_size=500, chunk_overlap=20)
    b = make_pipeline(settings, embeddings, fake_llm(["x"]), chunker="time")
    a.ingest(VIDEO_ID)
    b.ingest(VIDEO_ID)
    assert index_path(settings, VIDEO_ID, a.config) != index_path(settings, VIDEO_ID, b.config)


def test_ask_qa_returns_answer_sources_and_contexts(settings, embeddings, fake_llm):
    p = make_pipeline(settings, embeddings, fake_llm(["A producer writes events [1]."]), k=3)
    p.ingest(VIDEO_ID)
    result = p.ask("What is a producer?", [VIDEO_ID])
    assert result.route == "qa"
    assert result.answer == "A producer writes events [1]."
    assert len(result.sources) == len(result.contexts) == 3
    assert result.sources[0].url.startswith(f"https://youtu.be/{VIDEO_ID}?t=")
    assert result.sources[0].n == 1


def test_history_triggers_question_rewrite(settings, embeddings, fake_llm):
    llm = fake_llm(["How many replicas does a Kafka partition have?", "Three [1]."])
    p = make_pipeline(settings, embeddings, llm)
    p.ingest(VIDEO_ID)
    history = [
        {"role": "user", "content": "What is replication?"},
        {"role": "assistant", "content": "Copies of partitions."},
    ]
    result = p.ask("how many of them?", [VIDEO_ID], history)
    assert result.question == "How many replicas does a Kafka partition have?"
    assert result.answer == "Three [1]."


def test_summary_route_uses_map_reduce(settings, embeddings, fake_llm):
    # map_batch_chars small enough to force several map calls; the last response is the reduce.
    p = make_pipeline(settings, embeddings, fake_llm(["unused"]), map_batch_chars=1500)
    p.ingest(VIDEO_ID)
    chunks = dedupe_overlap(p.retriever([VIDEO_ID]).chunks)
    n_batches = len(batch_chunks(chunks, 1500))
    p.llm = fake_llm([f"- note {i}" for i in range(n_batches)] + ["Final summary."])
    result = p.ask("Summarize the video", [VIDEO_ID])
    assert result.route == "summary"
    assert result.answer == "Final summary."
    assert len(result.contexts) == len(result.sources) == n_batches > 1


def test_router_off_uses_normal_retrieval(settings, embeddings, fake_llm):
    p = make_pipeline(settings, embeddings, fake_llm(["ok"]), router=False)
    p.ingest(VIDEO_ID)
    assert p.ask("Summarize the video", [VIDEO_ID]).route == "qa"


def test_stream_yields_tokens(settings, embeddings, fake_llm):
    p = make_pipeline(settings, embeddings, fake_llm(["streamed answer"]))
    p.ingest(VIDEO_ID)
    prepared = p.prepare("What is a topic?", [VIDEO_ID])
    assert "".join(prepared.stream()) == "streamed answer"
