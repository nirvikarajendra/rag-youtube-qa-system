from ytrag.chunking import char_chunks, chunk_transcript, time_window_chunks
from ytrag.config import RAGConfig
from ytrag.youtube import Segment, Transcript


def test_time_windows_respect_window_and_keep_order(transcript):
    docs = time_window_chunks(transcript, window_seconds=60, max_chars=10_000, overlap_segments=1)
    assert len(docs) > 1
    starts = [d.metadata["start"] for d in docs]
    assert starts == sorted(starts)
    for d in docs:
        # 60s window + one carried-over 10s segment; each segment lasts 9.5s.
        assert d.metadata["end"] - d.metadata["start"] <= 60 + 10 + 9.5
        assert d.metadata["url"].endswith(f"?t={int(d.metadata['start'])}")
    # Chunks cover the whole transcript.
    assert docs[0].metadata["start"] == 0
    assert docs[-1].metadata["end"] == transcript.segments[-1].end


def test_time_windows_overlap_one_segment(transcript):
    docs = time_window_chunks(transcript, window_seconds=60, max_chars=10_000, overlap_segments=1)
    first_last_seg = docs[0].page_content.split(") ")[-1]
    assert docs[1].page_content.startswith(first_last_seg)


def test_time_windows_respect_max_chars(transcript):
    docs = time_window_chunks(transcript, window_seconds=10_000, max_chars=300, overlap_segments=0)
    assert all(len(d.page_content) <= 300 for d in docs)
    joined = " ".join(d.page_content for d in docs)
    assert joined == " ".join(s.text for s in transcript.segments)


def test_time_windows_no_duplicate_chunk_for_oversized_segment():
    segs = [Segment("short one", 0, 1), Segment("x" * 500, 1, 1), Segment("short two", 2, 1)]
    t = Transcript("abcdefghijk", "t", "en", segs)
    docs = time_window_chunks(t, window_seconds=60, max_chars=100, overlap_segments=1)
    texts = [d.page_content for d in docs]
    assert len(texts) == len(set(texts))


def test_char_chunks_map_back_to_timestamps(transcript):
    docs = char_chunks(transcript, chunk_size=200, chunk_overlap=20)
    assert len(docs) > 5
    for d in docs:
        start, end = d.metadata["start"], d.metadata["end"]
        assert start <= end
        # The chunk's first words must come from the segment its timestamp points at.
        seg = next(s for s in transcript.segments if s.start == start)
        first_word = d.page_content.split()[0]
        assert first_word in seg.text


def test_chunk_ids_unique(transcript):
    for cfg in (RAGConfig(chunker="char", chunk_size=300), RAGConfig(chunker="time")):
        docs = chunk_transcript(transcript, cfg)
        ids = [d.metadata["chunk_id"] for d in docs]
        assert len(ids) == len(set(ids))
        assert all(d.metadata["video_id"] == transcript.video_id for d in docs)
