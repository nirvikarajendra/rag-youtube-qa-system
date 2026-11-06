"""Command-line interface.

    python cli.py ingest https://youtu.be/B5j3uNBH8X4
    python cli.py videos
    python cli.py ask B5j3uNBH8X4 "What is a producer?"
    python cli.py chat B5j3uNBH8X4 [more video ids...]
"""

from __future__ import annotations

import argparse
import logging
import sys

from ytrag import RAGConfig, RAGPipeline
from ytrag.youtube import TranscriptError, extract_video_id


def print_answer(result) -> None:
    print(f"\n{result.answer}\n")
    if result.question:
        print(f"(route: {result.route} | searched: {result.question})")
    for s in result.sources:
        print(f"  [{s.n}] {s.title} @ {s.timestamp}  {s.url}")
    print()


def cmd_ingest(pipeline: RAGPipeline, args) -> None:
    for url in args.urls:
        try:
            info = pipeline.ingest(url, force=args.force)
            print(f"Indexed {info['video_id']}  '{info['title']}'  ({info['chunks']} chunks)")
        except (TranscriptError, ValueError) as exc:
            print(f"Skipped {url}: {exc}", file=sys.stderr)


def cmd_videos(pipeline: RAGPipeline, args) -> None:
    videos = pipeline.list_videos()
    if not videos:
        print("No videos yet. Run: python cli.py ingest <url>")
    for v in videos:
        print(f"{v['video_id']}  {v['duration'] / 60:5.1f} min  {v['title']}")


def cmd_ask(pipeline: RAGPipeline, args) -> None:
    print_answer(pipeline.ask(args.question, [extract_video_id(args.video)]))


def cmd_chat(pipeline: RAGPipeline, args) -> None:
    video_ids = [extract_video_id(v) for v in args.videos]
    history: list[dict] = []
    print("Chatting with", ", ".join(video_ids), "- type 'exit' to quit.\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        prepared = pipeline.prepare(question, video_ids, history)
        print("bot> ", end="", flush=True)
        answer = ""
        for token in prepared.stream():
            print(token, end="", flush=True)
            answer += token
        print()
        for s in prepared.sources:
            print(f"  [{s.n}] {s.timestamp}  {s.url}")
        print()
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with YouTube videos (RAG).")
    parser.add_argument("--retrieval", choices=["similarity", "mmr", "hybrid"], help="override retrieval mode")
    parser.add_argument("-k", type=int, help="override number of chunks retrieved")
    parser.add_argument("--rerank", action="store_true", help="enable LLM reranking")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="fetch, chunk, embed and save one or more videos")
    p.add_argument("urls", nargs="+")
    p.add_argument("--force", action="store_true", help="rebuild even if already indexed")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("videos", help="list ingested videos")
    p.set_defaults(func=cmd_videos)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("video")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive chat with memory")
    p.add_argument("videos", nargs="+")
    p.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    overrides = {}
    if args.retrieval:
        overrides["retrieval"] = args.retrieval
    if args.k:
        overrides["k"] = args.k
        overrides["fetch_k"] = max(args.k, RAGConfig().fetch_k)
    if args.rerank:
        overrides["rerank"] = True
    pipeline = RAGPipeline(config=RAGConfig().with_overrides(**overrides))
    args.func(pipeline, args)


if __name__ == "__main__":
    main()
