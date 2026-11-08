"""Run the RAG pipeline over the eval dataset and score it with RAGAS.

    python -m eval.run_eval --configs baseline --limit 5      # quick smoke test
    python -m eval.run_eval                                   # every experiment in configs.yaml
    python -m eval.run_eval --score-only eval/results/runs/baseline_20261027-120000.jsonl

Outputs (in eval/results/):
    runs/<config>_<stamp>.jsonl     pipeline outputs (question, answer, contexts) - rescorable
    scores_<config>_<stamp>.csv     per-question RAGAS scores
    summary.md                      latest scores for every config, overall and per question type
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# ragas 0.4 warns that its metric imports move in v1.0; the old paths still work.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from ytrag import RAGConfig, RAGPipeline, Settings
from ytrag.pipeline import default_embeddings, default_llm

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"
METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}

log = logging.getLogger("eval")


def load_dataset(path: Path, limit: int | None, types: list[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if types:
        rows = [r for r in rows if r["type"] in types]
    return rows[:limit] if limit else rows


def load_experiments(path: Path) -> dict[str, dict]:
    return yaml.safe_load(path.read_text())["experiments"]


def run_pipeline(name: str, overrides: dict, rows: list[dict], settings: Settings, llm, embeddings) -> list[dict]:
    overrides = {k: v for k, v in overrides.items() if k != "description"}
    pipeline = RAGPipeline(RAGConfig().with_overrides(**overrides), settings, llm=llm, embeddings=embeddings)
    for video_id in sorted({r["video_id"] for r in rows}):
        info = pipeline.ingest(video_id)
        log.info("[%s] %s: %d chunks", name, video_id, info["chunks"])

    records = []
    for i, row in enumerate(rows, start=1):
        start = time.perf_counter()
        record = {**row, "config": name, "response": "", "retrieved_contexts": [], "route": None, "error": None}
        try:
            result = pipeline.ask(row["question"], [row["video_id"]])
            record.update(
                response=result.answer,
                retrieved_contexts=result.contexts,
                route=result.route,
                source_starts=[s.start for s in result.sources],
            )
        except Exception as exc:  # keep going; failed rows are excluded from scoring
            record["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("[%s] %s failed: %s", name, row["id"], record["error"])
        record["latency_s"] = round(time.perf_counter() - start, 2)
        records.append(record)
        print(f"  [{name}] {i}/{len(rows)} {row['id']} ({record['latency_s']}s)")
    return records


def score(records: list[dict], settings: Settings, workers: int) -> pd.DataFrame:
    """RAGAS: faithfulness, answer relevancy, context precision, context recall."""
    ok =[r for r in records if not r["error"]]
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"] or [""],
                "reference": r["reference"],
            }
            for r in ok
        ]
    )
    # bypass_n: ask for answer relevancy's 3 questions in 3 calls; newer Gemini models reject n > 1.
    judge = LangchainLLMWrapper(default_llm(settings, settings.judge_model), bypass_n=True)
    embedder = LangchainEmbeddingsWrapper(default_embeddings(settings))
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=judge,
        embeddings=embedder,
        # Low concurrency + retries keeps Gemini free-tier rate limits happy.
        run_config=RunConfig(max_workers=workers, timeout=240, max_retries=10, max_wait=60),
        raise_exceptions=False,
    )
    scores = result.to_pandas().rename(columns=METRIC_COLUMNS)
    meta = pd.DataFrame(
        [{k: r[k] for k in ("id", "video_id", "type", "config", "route", "latency_s")} for r in ok]
    )
    return pd.concat([meta, scores[list(METRIC_COLUMNS.values())]], axis=1)


def write_summary(experiments: dict[str, dict]) -> Path:
    """Build summary.md from the newest scores CSV of each config."""
    latest: dict[str, Path] = {}
    for path in sorted(RESULTS_DIR.glob("scores_*.csv")):
        config = path.stem.removeprefix("scores_").rsplit("_", 1)[0]
        latest[config] = path  # sorted by stamp, so the last one wins
    if not latest:
        return RESULTS_DIR / "summary.md"

    metrics = list(METRIC_COLUMNS.values())
    order = [c for c in experiments if c in latest] + [c for c in latest if c not in experiments]
    frames = {c: pd.read_csv(latest[c]) for c in order}

    def table(df_by_config: dict[str, pd.DataFrame], first_col: str) -> list[str]:
        lines = [
            f"| {first_col} | n | " + " | ".join(m.replace("_", " ").title() for m in metrics) + " | Latency (s) |",
            "|---|---|" + "---|" * len(metrics) + "---|",
        ]
        for label, df in df_by_config.items():
            vals = " | ".join(f"{df[m].mean():.3f}" if df[m].notna().any() else "–" for m in metrics)
            lines.append(f"| {label} | {len(df)} | {vals} | {df['latency_s'].mean():.1f} |")
        return lines

    out = [f"# RAGAS results\n\n_Generated {datetime.now():%Y-%m-%d %H:%M}. Judge: {Settings().judge_model}._\n"]
    out.append("## Overall\n")
    out += table(frames, "Config")
    out.append("")
    for c in order:
        desc = experiments.get(c, {}).get("description", "")
        out.append(f"- **{c}**: {desc} (`{latest[c].name}`)")
    out.append("\n## By question type\n")
    for qtype in sorted({t for df in frames.values() for t in df["type"].unique()}):
        out.append(f"### {qtype}\n")
        out += table({c: df[df["type"] == qtype] for c, df in frames.items()}, "Config")
        out.append("")

    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(out) + "\n")
    return path


def save_scores(records: list[dict], name: str, stamp: str, settings: Settings, workers: int) -> None:
    failed = sum(1 for r in records if r["error"])
    if failed:
        print(f"  {failed} question(s) failed in the pipeline and are excluded from scoring")
    print(f"  scoring {len(records) - failed} answers with RAGAS...")
    df = score(records, settings, workers)
    path = RESULTS_DIR / f"scores_{name}_{stamp}.csv"
    df.to_csv(path, index=False)
    means = df[list(METRIC_COLUMNS.values())].mean().round(3).to_dict()
    print(f"  saved {path.relative_to(EVAL_DIR.parent)}  {means}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--experiments", type=Path, default=EVAL_DIR / "configs.yaml")
    parser.add_argument("--configs", nargs="+", help="experiment names (default: all)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--types", nargs="+", help="only these question types, e.g. fact summary")
    parser.add_argument("--workers", type=int, default=2, help="RAGAS concurrency (keep low on free tier)")
    parser.add_argument("--no-score", action="store_true", help="run the pipeline only, skip RAGAS")
    parser.add_argument("--score-only", type=Path, nargs="+", help="score saved run files instead")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    settings = Settings()
    settings.require_api_key()
    experiments = load_experiments(args.experiments)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        for run_file in args.score_only:
            records = [json.loads(line) for line in run_file.read_text().splitlines() if line.strip()]
            name, stamp = run_file.stem.rsplit("_", 1)
            print(f"\n== scoring {run_file.name}")
            save_scores(records, name, stamp, settings, args.workers)
        print(f"\nSummary: {write_summary(experiments)}")
        return

    names = args.configs or list(experiments)
    unknown = [n for n in names if n not in experiments]
    if unknown:
        parser.error(f"unknown config(s) {unknown}; available: {list(experiments)}")

    rows = load_dataset(args.dataset, args.limit, args.types)
    print(f"{len(rows)} questions x {len(names)} config(s)")
    llm = default_llm(settings)
    embeddings = default_embeddings(settings)

    for name in names:
        print(f"\n== {name}: {experiments[name].get('description', '')}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        records = run_pipeline(name, experiments[name], rows, settings, llm, embeddings)
        run_path = RUNS_DIR / f"{name}_{stamp}.jsonl"
        run_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
        print(f"  saved {run_path.relative_to(EVAL_DIR.parent)}")
        if not args.no_score:
            save_scores(records, name, stamp, settings, args.workers)

    print(f"\nSummary: {write_summary(experiments)}")


if __name__ == "__main__":
    main()
