from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import asyncpg
import numpy as np
from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.semantic_eval_utils import (
    fbeta_score,
    load_questions_with_embeddings,
    precision_recall,
)
from app.retrieval_shared import ann_query, init_pgvector


async def evaluate(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    top_k: int,
    concurrency: int,
    chunk_limit: int,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    column: str,
) -> float:
    """
    Bounded evaluation with worker pool.
    """
    q_list = list(questions)
    if not q_list:
        print("No questions to evaluate.")
        return 0.0

    q_iter = iter(q_list)
    lock = asyncio.Lock()

    async def next_item():
        async with lock:
            return next(q_iter, None)

    async def runner():
        f2s: List[float] = []
        ps: List[float] = []
        rs: List[float] = []

        while True:
            q = await next_item()
            if q is None:
                break

            preds = await ann_query(
                pool=pool,
                query_vec=q["embedding"],
                top_k=top_k,
                chunk_limit=chunk_limit,
                probes=probes,
                ef_search=ef_search,
                metric=metric,
                column=column,
            )
            pred_docs = [doc for doc, _ in preds]
            f2s.append(fbeta_score(q["gold_doc_ids"], pred_docs, beta=2.0))
            p, r = precision_recall(q["gold_doc_ids"], pred_docs)
            ps.append(p)
            rs.append(r)

        return f2s, ps, rs

    worker_n = max(1, concurrency)
    workers = [asyncio.create_task(runner()) for _ in range(worker_n)]
    parts = await asyncio.gather(*workers)

    f2_scores = [x for part in parts for x in part[0]]
    prec_scores = [x for part in parts for x in part[1]]
    rec_scores = [x for part in parts for x in part[2]]

    macro_f2 = float(np.mean(f2_scores)) if f2_scores else 0.0
    macro_prec = float(np.mean(prec_scores)) if prec_scores else 0.0
    macro_rec = float(np.mean(rec_scores)) if rec_scores else 0.0

    print(
        f"Chunk semantic (ANN/{metric}) @ {top_k}: macro F2={macro_f2:.4f} | "
        f"macro Precision={macro_prec:.4f} | macro Recall={macro_rec:.4f} "
        f"over {len(f2_scores)} questions"
    )
    return macro_f2


async def main_async(
    top_k: int,
    limit: int | None,
    concurrency: int,
    chunk_limit: int,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    emb_path: Path,
    meta_path: Path,
    column: str,
):
    questions = load_questions_with_embeddings(limit=limit, emb_path=emb_path, meta_path=meta_path)
    print(
        f"Evaluating ANN ({metric}) on {len(questions)} questions "
        f"(concurrency={concurrency}, chunk_limit={chunk_limit}, probes={probes}, ef_search={ef_search})."
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
        await evaluate(
            pool=pool,
            questions=questions,
            top_k=top_k,
            concurrency=concurrency,
            chunk_limit=chunk_limit,
            probes=probes,
            ef_search=ef_search,
            metric=metric,
            column=column,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(
        description="Chunk-level ANN semantic retrieval using pgvector index (async evaluation)."
    )
    parser.add_argument("--top-k", type=int, default=500, help="Top-K documents to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent DB queries; also limits DB pool size.",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=5000,
        help="Top chunks returned from ANN before doc aggregation (set high for recall).",
    )
    parser.add_argument(
        "--probes",
        type=int,
        default=None,
        help="Set ivfflat.probes for recall/speed tradeoff (requires IVF index).",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="Set hnsw.ef_search for recall/speed tradeoff (requires HNSW index).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Distance/score metric: cosine (<=>), l2 (<->), or ip (<#>).",
    )
    parser.add_argument(
        "--emb-path",
        type=Path,
        default=None,
        help="Path to question embedding .npy (default: data/train_embedding_bge_m3.npy).",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Path to question embedding meta JSON (default: data/train_embedding_meta.json).",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="embedding_bge_m3",
        help="Chunk embedding column to query (default: embedding_bge_m3).",
    )
    args = parser.parse_args()

    emb_path = args.emb_path or Path("data/train_embedding_bge_m3.npy")
    meta_path = args.meta_path or Path("data/train_embedding_meta.json")

    asyncio.run(
        main_async(
            top_k=args.top_k,
            limit=args.limit,
            concurrency=args.concurrency,
            chunk_limit=args.chunk_limit,
            probes=args.probes,
            ef_search=args.ef_search,
            metric=args.metric,
            emb_path=emb_path,
            meta_path=meta_path,
            column=args.column,
        )
    )


if __name__ == "__main__":
    main()
