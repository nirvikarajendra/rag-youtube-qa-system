"""Shared fixtures. Everything here is offline: fake LLM, fake embeddings, a synthetic transcript."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ytrag.config import Settings
from ytrag.youtube import Segment, Transcript

VIDEO_ID = "B5j3uNBH8X4"

LINES = [
    "A producer is an application that writes events into a Kafka cluster.",
    "The cluster is made of brokers and each broker has its own local storage.",
    "Consumers are programs that read data out of the cluster.",
    "ZooKeeper manages consensus and leader election for the brokers.",
    "A topic is a log of related events and new events go on the end.",
    "Topics are split into partitions so they can be spread across brokers.",
    "Each partition has a replication factor, usually three replicas.",
    "Consumer offsets are stored in a special internal topic.",
]


@pytest.fixture
def transcript() -> Transcript:
    # 8 lines repeated 5 times, 10 seconds per segment = 400 seconds of "video".
    segments = [
        Segment(text=f"{line} (part {rep})", start=float(i * 10), duration=9.5)
        for i, (rep, line) in enumerate((r, l) for r in range(5) for l in LINES)
    ]
    return Transcript(video_id=VIDEO_ID, title="Kafka Fundamentals", language="en", segments=segments)


@pytest.fixture
def settings(tmp_path: Path, transcript: Transcript) -> Settings:
    s = Settings(data_dir=tmp_path, embedding_model="fake-embedding")
    s.transcripts_dir.mkdir(parents=True)
    # Pre-seed the transcript cache so nothing hits YouTube.
    (s.transcripts_dir / f"{VIDEO_ID}.json").write_text(json.dumps(transcript.to_dict()))
    return s


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def fake_llm():
    def make(responses: list[str]) -> FakeListChatModel:
        return FakeListChatModel(responses=responses)

    return make
