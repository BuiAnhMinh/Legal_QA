from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Iterable, List

import asyncpg

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.semantic_eval_utils import load_questions_with_embeddings, tokenize_question
from app.retrieval_shared import (
    ann_query,
    bm25_query,
    hybrid_score_map,
    init_pgvector,
)


def _aggregate(scores: List[float], mode: str) -> float:
    if not scores:
        return 0.0
    if mode == "mean":
        return sum(scores) / len(scores)
    if mode == "min":
        return min(scores)
    return max(scores)


async def evaluate(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    bm25_top: int,
    dense_docs: int,
    dense_chunks: int,
    alpha: float,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    agg_mode: str,
    concurrency: int,
) -> List[Dict]:
    q_list = list(questions)
    if not q_list:
        return []

    q_iter = iter(q_list)
    lock = asyncio.Lock()

    async def next_item():
        async with lock:
            return next(q_iter, None)

    async def runner():
        out: List[Dict] = []
        while True:
            q = await next_item()
            if q is None:
                break

            tokens = tokenize_question(q["text"])
            bm25_pairs = await bm25_query(pool=pool, query_terms=tokens, top_k=bm25_top)
            dense_pairs = await ann_query(
                pool=pool,
                query_vec=q["embedding"],
                top_k=dense_docs,
                chunk_limit=dense_chunks,
                probes=probes,
                ef_search=ef_search,
                metric=metric,
            )

            score_map = hybrid_score_map(bm25_pairs, dense_pairs, alpha=alpha)
            gold_ids = sorted(list(q["gold_doc_ids"]))
            gold_scores = [(doc_id, score_map.get(doc_id, 0.0)) for doc_id in gold_ids]
            gold_scores.sort(key=lambda x: x[1], reverse=True)

            agg_score = _aggregate([s for _, s in gold_scores], agg_mode)
            out.append(
                {
                    "question_id": q["question_id"],
                    "text": q["text"],
                    "agg_score": agg_score,
                    "gold_scores": gold_scores,
                }
            )
        return out

    worker_n = max(1, concurrency)
    workers = [asyncio.create_task(runner()) for _ in range(worker_n)]
    parts = await asyncio.gather(*workers)
    merged = [item for part in parts for item in part]
    merged.sort(key=lambda x: x["agg_score"], reverse=True)
    return merged


async def main_async(
    bm25_top: int,
    dense_docs: int,
    dense_chunks: int,
    alpha: float,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    agg_mode: str,
    limit: int | None,
    concurrency: int,
    emb_path: Path,
    meta_path: Path,
    out_path: Path | None,
):
    questions = load_questions_with_embeddings(
        limit=limit,
        emb_path=emb_path,
        meta_path=meta_path,
    )

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=max(1, concurrency),
        init=init_pgvector,
    )

    try:
        ranked = await evaluate(
            pool=pool,
            questions=questions,
            bm25_top=bm25_top,
            dense_docs=dense_docs,
            dense_chunks=dense_chunks,
            alpha=alpha,
            probes=probes,
            ef_search=ef_search,
            metric=metric,
            agg_mode=agg_mode,
            concurrency=concurrency,
        )
    finally:
        await pool.close()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for idx, row in enumerate(ranked, start=1):
                payload = dict(row)
                payload["rank"] = idx
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"Wrote {len(ranked)} rows to {out_path}")
        return

    for idx, row in enumerate(ranked, start=1):
        print("=" * 80)
        print(f"Rank: {idx} | agg_score={row['agg_score']:.6f}")
        print(f"Question ID: {row['question_id']}")
        print(row["text"])
        print("Gold scores (doc_id, hybrid_score):")
        for doc_id, score in row["gold_scores"]:
            print(f"  {doc_id}: {score:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank questions by hybrid score of their gold docs."
    )
    parser.add_argument("--bm25-top", type=int, default=500, help="BM25 doc candidates.")
    parser.add_argument(
        "--dense-docs",
        type=int,
        default=500,
        help="Top docs from ANN aggregation.",
    )
    parser.add_argument(
        "--dense-chunks",
        type=int,
        default=2000,
        help="Top chunks from ANN before doc aggregation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 in hybrid score (dense weight = 1 - alpha).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Distance/score metric for ANN.",
    )
    parser.add_argument("--probes", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=None)
    parser.add_argument(
        "--agg",
        type=str,
        default="max",
        choices=["max", "mean", "min"],
        help="Aggregate gold scores for question ranking.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--emb-path",
        type=Path,
        default=None,
        help="Embedding .npy for dense mode (default: data/train_embedding_bge_m3.npy).",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Embedding meta JSON (default: data/train_embedding_meta.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSONL path.",
    )
    args = parser.parse_args()

    emb_path = args.emb_path or Path("data/train_embedding_bge_m3.npy")
    meta_path = args.meta_path or Path("data/train_embedding_meta.json")

    asyncio.run(
        main_async(
            bm25_top=args.bm25_top,
            dense_docs=args.dense_docs,
            dense_chunks=args.dense_chunks,
            alpha=args.alpha,
            probes=args.probes,
            ef_search=args.ef_search,
            metric=args.metric,
            agg_mode=args.agg,
            limit=args.limit,
            concurrency=args.concurrency,
            emb_path=emb_path,
            meta_path=meta_path,
            out_path=args.out,
        )
    )


if __name__ == "__main__":
    main()
