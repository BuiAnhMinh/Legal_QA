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
    tokenize_question,
)


async def _init_pgvector(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def jaccard_query(
    pool: asyncpg.pool.Pool,
    query_tokens: Sequence[str],
    query_vec: Sequence[float],
    top_k: int,
    chunk_limit: int,
    candidate_doc_limit: int,
) -> List[int]:
    """
    Hybrid Jaccard:
      - Use vector search to pick top chunks (fast, indexed).
      - Collapse to candidate docs, then compute Jaccard on tokens inside that set.
    """
    tokens = [t for t in query_tokens if t]
    if not tokens or query_vec is None:
        return []

    sql = """
    WITH params AS (
        SELECT
            ARRAY(SELECT DISTINCT qt FROM unnest($1::text[]) qt) AS q_tokens,
            $2::vector AS qvec,
            $3::int AS chunk_limit,
            $4::int AS cand_docs,
            $5::int AS out_docs
    ),

    -- 1) Vector prefilter: score all chunks and take top N by distance
    scored_chunks AS (
        SELECT
            ac.doc_id,
            (ac.embedding_bge_m3 <=> p.qvec) AS dist,
            ROW_NUMBER() OVER (ORDER BY ac.embedding_bge_m3 <=> p.qvec ASC) AS rn
        FROM article_chunks ac, params p
        WHERE ac.embedding_bge_m3 IS NOT NULL
          AND ac.doc_id IS NOT NULL
    ),
    top_chunks AS (
        SELECT doc_id, dist
        FROM scored_chunks, params p
        WHERE rn <= p.chunk_limit
    ),

    -- 2) Candidate docs by best chunk distance
    cand_docs AS (
        SELECT doc_id, ROW_NUMBER() OVER (ORDER BY MIN(dist)) AS rn
        FROM top_chunks
        GROUP BY doc_id
    ),
    filtered_docs AS (
        SELECT doc_id
        FROM cand_docs, params p
        WHERE rn <= p.cand_docs
    ),

    -- 3) Token Jaccard within candidate docs
    chunk_tokens AS (
        SELECT
            ac.doc_id,
            ARRAY(SELECT DISTINCT t FROM unnest(ac.token) t) AS c_tokens
        FROM article_chunks ac, params p
        WHERE ac.doc_id IN (SELECT doc_id FROM filtered_docs)
          AND ac.token IS NOT NULL
          AND ac.token && p.q_tokens
    ),
    scored AS (
        SELECT
            ct.doc_id,
            (
                SELECT COUNT(*) FROM unnest(ct.c_tokens) t WHERE t = ANY (p.q_tokens)
            ) AS inter,
            cardinality(ct.c_tokens) AS c_size,
            cardinality(p.q_tokens) AS q_size
        FROM chunk_tokens ct, params p
    ),
    doc_scores AS (
        SELECT
            doc_id,
            MAX(
                CASE
                    WHEN (q_size + c_size - inter) = 0 THEN 0
                    ELSE inter::float / (q_size + c_size - inter)
                END
            ) AS score
        FROM scored
        GROUP BY doc_id
    ),
    ranked AS (
        SELECT doc_id, score, ROW_NUMBER() OVER (ORDER BY score DESC) AS rn
        FROM doc_scores
    )
    SELECT doc_id
    FROM ranked, params p
    WHERE rn <= p.out_docs;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            sql,
            tokens,
            query_vec,
            chunk_limit,
            candidate_doc_limit,
            top_k,
        )
    return [int(r["doc_id"]) for r in rows]


async def evaluate(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    top_k: int,
    concurrency: int,
    chunk_limit: int,
    candidate_doc_limit: int,
) -> float:
    """
    Bounded evaluation: a pool of worker tasks pulls from a shared iterator.
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

            q_tokens = tokenize_question(q["text"])
            preds = await jaccard_query(
                pool=pool,
                query_tokens=q_tokens,
                query_vec=q["embedding"],
                top_k=top_k,
                chunk_limit=chunk_limit,
                candidate_doc_limit=candidate_doc_limit,
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
        f"Chunk lexical (Jaccard tokens) @ {top_k}: macro F2={macro_f2:.4f} | "
        f"macro Precision={macro_prec:.4f} | macro Recall={macro_rec:.4f} "
        f"over {len(f2_scores)} questions"
    )
    return macro_f2


async def main_async(
    top_k: int,
    limit: int | None,
    concurrency: int,
    chunk_limit: int,
    candidate_doc_limit: int,
    emb_path: Path,
    meta_path: Path,
):
    questions = load_questions_with_embeddings(limit=limit, emb_path=emb_path, meta_path=meta_path)
    print(
        f"Evaluating Jaccard on {len(questions)} questions "
        f"(concurrency={concurrency}, chunk_limit={chunk_limit}, cand_docs={candidate_doc_limit})."
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
            chunk_limit=chunk_limit,
            candidate_doc_limit=candidate_doc_limit,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(
        description="Chunk-level Jaccard token similarity with vector-prefiltered candidates (async)."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K documents to evaluate.")
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
        default=300,
        help="Top chunks (by cosine) used to build Jaccard doc candidates.",
    )
    parser.add_argument(
        "--candidate-doc-limit",
        type=int,
        default=150,
        help="Top docs (from chunk prefilter) to score with Jaccard.",
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
            chunk_limit=args.chunk_limit,
            candidate_doc_limit=args.candidate_doc_limit,
            emb_path=emb_path,
            meta_path=meta_path,
        )
    )


if __name__ == "__main__":
    main()
