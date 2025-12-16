from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.semantic_eval_utils import (
    fbeta_score,
    load_questions_with_embeddings,
    precision_recall,
)


async def _init_pgvector(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def ann_query(
    pool: asyncpg.pool.Pool,
    query_vec: Sequence[float],
    top_k: int,
    probes: int | None = None,
    metric: str = "cosine",
) -> List[int]:
    """
    ANN via pgvector index on article_chunks.embedding_bge_m3.
    metric: 'cosine' (<=>) or 'l2' (<->) or 'ip' (<#>)
    Returns top_k doc_ids by best chunk score (max for ip, min for dist).
    """
    if query_vec is None:
        return []
    if hasattr(query_vec, "size") and query_vec.size == 0:
        return []
    if not hasattr(query_vec, "size") and len(query_vec) == 0:
        return []

    q_list = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)

    op = {
        "cosine": "<=>",
        "l2": "<->",
        "ip": "<#>",
    }.get(metric, "<=>")

    order = "ASC" if metric in ("cosine", "l2") else "DESC"

    sql = f"""
    WITH top_chunks AS (
        SELECT
            ac.doc_id,
            ac.embedding_bge_m3 {op} $1::vector AS score
        FROM article_chunks ac
        WHERE ac.embedding_bge_m3 IS NOT NULL
          AND ac.doc_id IS NOT NULL
        ORDER BY ac.embedding_bge_m3 {op} $1::vector {order}
        LIMIT $2
    ),
    doc_scores AS (
        SELECT doc_id, MIN(score) AS best_score
        FROM top_chunks
        GROUP BY doc_id
    )
    SELECT doc_id
    FROM doc_scores
    ORDER BY best_score {order}
    LIMIT $3;
    """

    async with pool.acquire() as conn:
        if probes is not None:
            await conn.execute("SET ivfflat.probes = $1;", probes)
        rows = await conn.fetch(sql, q_list, top_k, top_k)

    return [int(r["doc_id"]) for r in rows]


async def evaluate(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    top_k: int,
    concurrency: int,
    probes: int | None,
    metric: str,
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
                probes=probes,
                metric=metric,
            )
            f2s.append(fbeta_score(q["gold_doc_ids"], preds, beta=2.0))
            p, r = precision_recall(q["gold_doc_ids"], preds)
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
    probes: int | None,
    metric: str,
    emb_path: Path,
    meta_path: Path,
):
    questions = load_questions_with_embeddings(limit=limit, emb_path=emb_path, meta_path=meta_path)
    print(
        f"Evaluating ANN ({metric}) on {len(questions)} questions "
        f"(concurrency={concurrency}, probes={probes})."
    )

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=max(1, concurrency),
        init=_init_pgvector,
    )

    try:
        await evaluate(
            pool=pool,
            questions=questions,
            top_k=top_k,
            concurrency=concurrency,
            probes=probes,
            metric=metric,
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
        "--probes",
        type=int,
        default=None,
        help="Set ivfflat.probes for recall/speed tradeoff (requires IVF index).",
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
    args = parser.parse_args()

    emb_path = args.emb_path or Path("data/train_embedding_bge_m3.npy")
    meta_path = args.meta_path or Path("data/train_embedding_meta.json")

    asyncio.run(
        main_async(
            top_k=args.top_k,
            limit=args.limit,
            concurrency=args.concurrency,
            probes=args.probes,
            metric=args.metric,
            emb_path=emb_path,
            meta_path=meta_path,
        )
    )


if __name__ == "__main__":
    main()
