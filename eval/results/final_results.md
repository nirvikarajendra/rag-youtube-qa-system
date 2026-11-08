# Final evaluation results

Run on 2026-11-06 with the app's default setup (`full`: 60-second chunks, hybrid BM25 + FAISS, k=4,
summary router on). Generator and RAGAS judge: `gemini-3.5-flash-lite`.

Dataset: 21 hand-written questions with reference answers on one 24-minute video
(Apache Kafka Fundamentals): 13 fact, 4 multi-chunk, 2 summary, 2 unanswerable.

## RAGAS scores (19 answerable questions)

| Question type | n | Faithfulness | Answer relevancy | Context precision | Context recall | Latency (s) |
|---|---|---|---|---|---|---|
| fact | 13 | 1.000 | 0.841 | 0.923 | 0.846 | 1.9 |
| multi_chunk | 4 | 1.000 | 0.833 | 0.875 | 0.917 | 1.7 |
| summary | 2 | 1.000 | 0.804 | 0.167* | 1.000 | 3.4 |
| **All answerable** | **19** | **1.000** | **0.835** | **0.833** | **0.877** | **2.0** |
| Normal questions only (fact + multi_chunk) | 17 | 1.000 | 0.839 | 0.912 | 0.863 | 1.8 |

\* For summary questions the contexts are the four map-step section notes. Context precision asks
whether each note alone leads to the whole-video reference, which a single section never does, so
this number is not meaningful for summaries. Report precision on the 17 normal questions (0.912).

## Baseline comparison (17 normal questions, same judge)

`baseline` = basic RAG: 500-character chunks, top-2 vector similarity, no router.

| Metric | Baseline | Full (app) | Change |
|---|---|---|---|
| Context recall | 0.745 | 0.863 | +0.118 |
| Context precision | 0.765 | 0.912 | +0.147 |
| Faithfulness | 0.951 | 1.000 | +0.049 |
| Answer relevancy | 0.851 | 0.839 | -0.012 (noise) |
| Latency (s) | 1.7 | 1.8 | +0.2 |

Biggest gains: kafka-17 recall 0.00 → 1.00, kafka-07 recall 0.50 → 1.00 and precision 0.00 → 1.00,
kafka-04 recall 0.50 → 1.00. One regression: kafka-14 recall 0.67 → 0.33 (answer spread across the video).

## Second video (generalization check)

Run on 2026-11-07, same `full` config and judge, on a 9-minute AI explainer
(`OnXVUXktKxQ`, 11 chunks): 10 hand-written questions, 6 fact, 2 multi-chunk, 1 summary, 1 unanswerable.
Questions are in `eval/dataset_ai_video.jsonl`.

| Question type | n | Faithfulness | Answer relevancy | Context precision | Context recall | Latency (s) |
|---|---|---|---|---|---|---|
| fact | 6 | 0.972 | 0.837 | 0.722 | 1.000 | 2.3* |
| multi_chunk | 2 | 0.857 | 0.866 | 1.000 | 1.000 | 1.3 |
| summary | 1 | 1.000 | 0.820 | 1.000† | 1.000 | 3.3 |
| **All answerable** | **9** | **0.949** | **0.841** | **0.792** (normal only) | **1.000** | 2.2 |

\* Includes one 8.9 s outlier (ai-04, a rate-limit retry); without it, fact latency is 1.0 s.
† Only 2 map notes for this short video, so each covers half of it; context precision is still not
meaningful for summaries (Kafka's 4 notes gave 0.167).

Notes:
- Recall is 1.000 partly because the video is short: k=4 of 11 chunks is about a third of the video
  (Kafka: 4 of 29).
- ai-07 (0.71 faithfulness) is a real error: the answer listed "misbehavior mostly happened with
  guardrails off in testing" as a limitation of guardrails.
- ai-05 (0.83) looks like judge strictness; every claim is in the retrieved text.
- Lower precision (ai-08 0.33, ai-01 and ai-05 0.50) comes from ranking: the one useful chunk
  was not ranked first.

### Both videos combined

| | Kafka | AI video | Combined |
|---|---|---|---|
| Answerable questions | 19 | 9 | 28 |
| Faithfulness | 1.000 | 0.949 | 0.984 |
| Answer relevancy | 0.835 | 0.841 | 0.837 |
| Context precision (normal questions) | 0.912 (17) | 0.792 (8) | 0.874 (25) |
| Context recall | 0.877 | 1.000 | 0.917 |
| Unanswerable → "I don't know" | 2 / 2 | 1 / 1 | 3 / 3 |
| Router correct | 19 / 19 | 9 / 9 | 28 / 28 |

## Checks outside RAGAS

| Check | Result |
|---|---|
| Unanswerable questions answered with "I don't know based on this video." | 2 / 2 |
| Router sent summary questions to map-reduce and the rest to retrieval | 19 / 19 |
| Answers with no faithfulness, relevancy, precision or recall score (NaN) | 0 |

## Weakest questions (context recall)

| Question | Recall | Why |
|---|---|---|
| kafka-14: Which programming languages can Kafka clients be written in? | 0.33 | the community libraries, REST Proxy and CLI tools are in a chunk outside the top 4 |
| kafka-02: What is a producer? | 0.50 | the acknowledgement detail is in a different chunk |
| kafka-03: What are brokers? | 0.50 | part of the reference (partitions, pub/sub) is in a later chunk |

## Caveats

- Small test set: 31 questions on two videos, so one question moves a per-video average by 5 to 10 points.
- The same model family generates and judges, which can inflate scores.
- `gemini-3.5-flash-lite` ignores the temperature setting (fixed sampling), so reruns can vary slightly.
- Follow-up rewriting and multi-video selection are not covered by this dataset.

## Per-question scores

`final_scores.csv` has every scored question (45 rows): `full` on both videos (28) and `baseline` on
Kafka (17), with one row per question and config. Raw answers and retrieved chunks are written to
`runs/` when the eval runs (not committed).
