"""Draft extra eval questions with RAGAS TestsetGenerator.

    python -m eval.generate_testset <video url or id> [...] --size 10

Writes eval/dataset_draft.jsonl. The questions and reference answers are LLM-written drafts:
check every one against the video, fix or delete bad ones, then copy the good ones into
eval/dataset.jsonl. Only hand-verified rows should be used for reported scores.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

# ragas 0.4 warns that some imports move in v1.0; the old paths still work.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

from ytrag import Settings
from ytrag.chunking import time_window_chunks
from ytrag.pipeline import default_embeddings, default_llm
from ytrag.youtube import extract_video_id, load_transcript

EVAL_DIR = Path(__file__).resolve().parent
TYPE_MAP = {
    "single_hop_specific_query_synthesizer": "fact",
    "multi_hop_abstract_query_synthesizer": "multi_chunk",
    "multi_hop_specific_query_synthesizer": "multi_chunk",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--size", type=int, default=10, help="questions per video")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "dataset_draft.jsonl")
    args = parser.parse_args()

    settings = Settings()
    generator = TestsetGenerator(
        llm=LangchainLLMWrapper(default_llm(settings)),
        embedding_model=LangchainEmbeddingsWrapper(default_embeddings(settings)),
    )

    rows = []
    for raw in args.videos:
        video_id = extract_video_id(raw)
        transcript = load_transcript(video_id, settings.transcripts_dir, settings.transcript_languages)
        # RAGAS needs reasonably long documents, so use ~5-minute sections here.
        docs = time_window_chunks(transcript, window_seconds=300, max_chars=6000, overlap_segments=0)
        print(f"{video_id}: generating {args.size} questions from {len(docs)} sections...")
        testset = generator.generate_with_langchain_docs(docs, testset_size=args.size)
        for i, rec in enumerate(testset.to_pandas().to_dict("records"), start=1):
            rows.append(
                {
                    "id": f"{video_id}-gen-{i:02d}",
                    "video_id": video_id,
                    "type": TYPE_MAP.get(rec.get("synthesizer_name", ""), "fact"),
                    "question": rec["user_input"],
                    "reference": rec["reference"],
                    "verified": False,
                }
            )

    with args.out.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Appended {len(rows)} draft questions to {args.out}. Review them before using.")


if __name__ == "__main__":
    main()
