from __future__ import annotations

import argparse
import asyncio
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import asyncpg
import numpy as np
from underthesea import word_tokenize

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.data_loader import load_train_data


def tokenize_question(text: str) -> List[str]:
    tok_str = word_tokenize(text or "", format="text")
    return [t.lower() for t in tok_str.split() if t]


def fbeta_score(gold: Set[int], pred: Sequence[int], beta: float = 2.0) -> float:
    gold = set(gold or [])
    pred_set = set(pred)
    if not gold and not pred_set:
        return 1.0

    tp = len(gold & pred_set)
    fp = len(pred_set - gold)
    fn = len(gold - pred_set)
    if tp == 0:
        return 0.0

    beta2 = beta * beta
    return (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp)


def precision_recall(gold: Set[int], pred: Sequence[int]) -> tuple[float, float]:
    gold = set(gold or [])
    pred_set = set(pred)
    tp = len(gold & pred_set)
    fp = len(pred_set - gold)
    fn = len(gold - pred_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def _dedup(tokens: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


async def fetch_chunk_collection_stats(pool: asyncpg.pool.Pool) -> Tuple[float, float]:
    sql = """
    SELECT
        COUNT(*)::float AS n_docs,
        COALESCE(AVG(cardinality(token)), 0)::float AS avg_doc_len
    FROM article_chunks
    WHERE doc_id IS NOT NULL;
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql)

    n_docs = float(row["n_docs"] or 0.0)
    avg_doc_len = float(row["avg_doc_len"] or 0.0)
    return n_docs, avg_doc_len


async def chunk_bm25_query(
    pool: asyncpg.pool.Pool,
    query_terms: Sequence[str],
    top_k: int,
    collection_stats: Tuple[float, float],
) -> List[int]:
    tokens = _dedup([t for t in query_terms if t])
    n_docs, avg_doc_len = collection_stats
    if not tokens or n_docs <= 0 or avg_doc_len <= 0:
        return []

    sql = """
    WITH params AS (
        SELECT
            $1::text[] AS tokens,
            $2::float AS n_docs,
            $3::float AS avg_doc_len,
            1.2::float AS k1,
            0.75::float AS b
    ),
    q_tokens AS (
        SELECT DISTINCT token
        FROM params, unnest(tokens) AS token
    ),
    tf AS (
        SELECT
            ac.id AS chunk_id,
            ac.doc_id AS doc_id,
            qt.token AS token,
            COUNT(*) AS tf,
            cardinality(ac.token) AS doc_len
        FROM article_chunks ac
        JOIN q_tokens qt ON TRUE
        JOIN LATERAL unnest(ac.token) AS ct(token) ON ct.token = qt.token
        WHERE ac.doc_id IS NOT NULL
        GROUP BY ac.id, ac.doc_id, qt.token, cardinality(ac.token)
    ),
    df AS (
        SELECT token, COUNT(DISTINCT chunk_id)::float AS df
        FROM tf
        GROUP BY token
    ),
    chunk_scores AS (
        SELECT
            tf.chunk_id,
            tf.doc_id,
            SUM(
                ln((p.n_docs - d.df + 0.5) / (d.df + 0.5) + 1) *
                (tf.tf * (p.k1 + 1)) /
                (tf.tf + p.k1 * (1 - p.b + p.b * tf.doc_len / p.avg_doc_len))
            ) AS score
        FROM tf
        JOIN df d ON tf.token = d.token
        JOIN params p ON TRUE
        GROUP BY tf.chunk_id, tf.doc_id
    ),
    doc_scores AS (
        SELECT doc_id, MAX(score) AS score
        FROM chunk_scores
        GROUP BY doc_id
    )
    SELECT doc_id
    FROM doc_scores
    ORDER BY score DESC
    LIMIT $4;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tokens, n_docs, avg_doc_len, top_k)
    return [int(r["doc_id"]) for r in rows]


async def evaluate_chunk_bm25(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    top_k: int,
    concurrency: int,
    collection_stats: Tuple[float, float],
) -> float:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def worker(q: Dict) -> tuple[float, float, float]:
        async with sem:
            q_tokens = await asyncio.to_thread(tokenize_question, q["text"])
            preds = await chunk_bm25_query(
                pool=pool,
                query_terms=q_tokens,
                top_k=top_k,
                collection_stats=collection_stats,
            )
            f2 = fbeta_score(q["gold_doc_ids"], preds, beta=2.0)
            prec, rec = precision_recall(q["gold_doc_ids"], preds)
            return f2, prec, rec

    tasks = [asyncio.create_task(worker(q)) for q in questions]
    results = await asyncio.gather(*tasks)
    if not results:
        print("No questions to evaluate.")
        return 0.0

    f2_scores, prec_scores, rec_scores = zip(*results)
    macro_f2 = float(np.mean(f2_scores))
    macro_prec = float(np.mean(prec_scores))
    macro_rec = float(np.mean(rec_scores))

    print(
        f"Chunk BM25 (per-chunk scored, max-doc aggregation) @ {top_k}: "
        f"macro F2={macro_f2:.4f} | macro Precision={macro_prec:.4f} | "
        f"macro Recall={macro_rec:.4f} over {len(results)} questions"
    )
    return macro_f2


async def main_async(top_k: int, limit: int | None, concurrency: int):
    concurrency = max(1, concurrency)
    questions = load_train_data()
    if limit:
        questions = questions[:limit]
    print(f"Evaluating on {len(questions)} questions (concurrency={concurrency}).")

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=max(1, concurrency),
    )

    try:
        collection_stats = await fetch_chunk_collection_stats(pool)
        n_docs, avg_doc_len = collection_stats
        print(f"Chunk collection stats: n_chunks={int(n_docs)}, avg_len={avg_doc_len:.2f}")
        if n_docs <= 0 or avg_doc_len <= 0:
            print("No chunk stats available; aborting.")
            return

        await evaluate_chunk_bm25(
            pool=pool,
            questions=questions,
            top_k=top_k,
            concurrency=concurrency,
            collection_stats=collection_stats,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(
        description="Chunk-level BM25 retrieval (DB-side) with async/threaded question fan-out."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K documents to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent BM25 queries; also limits DB pool size.",
    )
    args = parser.parse_args()

    asyncio.run(main_async(top_k=args.top_k, limit=args.limit, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
