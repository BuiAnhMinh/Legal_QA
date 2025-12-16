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


async def bm25_query(
    pool: asyncpg.pool.Pool,
    query_terms: Sequence[str],
    top_k: int,
) -> List[tuple[int, float]]:
    """
    BM25 over precomputed stats (article_term_freq, term_stats, article_stats, collection_stats).
    Returns list of (doc_id, score).
    """
    tokens = [t for t in query_terms if t]
    if not tokens:
        return []

    sql = """
    WITH params AS (
        SELECT
            $1::text[] AS tokens,
            1.2::float AS k1,
            0.75::float AS b
    ),
    scored AS (
        SELECT
            atf.article_id AS doc_id,
            SUM(
                ts.idf *
                (atf.tf * (p.k1 + 1)) /
                (atf.tf + p.k1 * (1 - p.b + p.b * ast.doc_len / cs.avg_doc_len))
            ) AS score
        FROM article_term_freq atf
        JOIN term_stats ts
          ON ts.token = atf.token
        JOIN article_stats ast
          ON ast.article_id = atf.article_id
        JOIN collection_stats cs
          ON cs.id = 1
        JOIN params p ON TRUE
        WHERE atf.token = ANY (p.tokens)
        GROUP BY atf.article_id
    )
    SELECT doc_id, score
    FROM scored
    ORDER BY score DESC
    LIMIT $2;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tokens, top_k)
    return [(int(r["doc_id"]), float(r["score"])) for r in rows]


async def ann_query(
    pool: asyncpg.pool.Pool,
    query_vec: Sequence[float],
    top_k: int,
    chunk_limit: int,
    probes: int | None = None,
    ef_search: int | None = None,
    metric: str = "cosine",
) -> List[tuple[int, float]]:
    """
    ANN over chunk embeddings; returns per-doc best chunk score as (doc_id, score).
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
    SELECT doc_id, best_score AS score
    FROM doc_scores
    ORDER BY best_score {order}
    LIMIT $3;
    """

    async with pool.acquire() as conn:
        if probes is not None:
            await conn.execute(f"SET ivfflat.probes = {int(probes)};")
        if ef_search is not None:
            await conn.execute(f"SET hnsw.ef_search = {int(ef_search)};")
        rows = await conn.fetch(sql, q_list, chunk_limit, top_k)

    return [(int(r["doc_id"]), float(r["score"])) for r in rows]


def _min_max_norm(pairs: List[tuple[int, float]]) -> Dict[int, float]:
    if not pairs:
        return {}
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {doc: 1.0 for doc, _ in pairs}
    return {doc: (score - lo) / (hi - lo) for doc, score in pairs}


def hybrid_merge(
    bm25_pairs: List[tuple[int, float]],
    dense_pairs: List[tuple[int, float]],
    alpha: float,
) -> List[int]:
    bm25_norm = _min_max_norm(bm25_pairs)
    dense_norm = _min_max_norm(dense_pairs)
    doc_ids = set(bm25_norm) | set(dense_norm)

    combined = []
    for doc in doc_ids:
        b = bm25_norm.get(doc, 0.0)
        d = dense_norm.get(doc, 0.0)
        combined.append((doc, alpha * b + (1 - alpha) * d))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in combined]


async def evaluate(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    top_k: int,
    bm25_top: int,
    dense_chunks: int,
    alpha: float,
    concurrency: int,
    probes: int | None,
    ef_search: int | None,
    metric: str,
) -> float:
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

            bm25_pairs = await bm25_query(pool=pool, query_terms=tokenize_question(q["text"]), top_k=bm25_top)
            dense_pairs = await ann_query(
                pool=pool,
                query_vec=q["embedding"],
                top_k=top_k,
                chunk_limit=dense_chunks,
                probes=probes,
                ef_search=ef_search,
                metric=metric,
            )
            hybrid_docs = hybrid_merge(bm25_pairs, dense_pairs, alpha=alpha)[:top_k]

            f2s.append(fbeta_score(q["gold_doc_ids"], hybrid_docs, beta=2.0))
            p, r = precision_recall(q["gold_doc_ids"], hybrid_docs)
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
        f"Hybrid BM25+ANN @ {top_k}: alpha={alpha:.2f}, bm25_top={bm25_top}, dense_chunks={dense_chunks}, "
        f"macro F2={macro_f2:.4f} | macro Precision={macro_prec:.4f} | macro Recall={macro_rec:.4f} "
        f"over {len(f2_scores)} questions"
    )
    return macro_f2


async def main_async(
    top_k: int,
    limit: int | None,
    bm25_top: int,
    dense_chunks: int,
    alpha: float,
    concurrency: int,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    emb_path: Path,
    meta_path: Path,
):
    questions = load_questions_with_embeddings(limit=limit, emb_path=emb_path, meta_path=meta_path)
    print(
        f"Evaluating Hybrid BM25+ANN on {len(questions)} questions "
        f"(top_k={top_k}, bm25_top={bm25_top}, dense_chunks={dense_chunks}, alpha={alpha}, "
        f"concurrency={concurrency}, probes={probes}, ef_search={ef_search}, metric={metric})."
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
            bm25_top=bm25_top,
            dense_chunks=dense_chunks,
            alpha=alpha,
            concurrency=concurrency,
            probes=probes,
            ef_search=ef_search,
            metric=metric,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + ANN (chunk-level) retrieval with async evaluation."
    )
    parser.add_argument("--top-k", type=int, default=500, help="Final top-K documents to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions.")
    parser.add_argument(
        "--bm25-top",
        type=int,
        default=500,
        help="BM25 doc candidates.",
    )
    parser.add_argument(
        "--dense-chunks",
        type=int,
        default=2000,
        help="ANN chunk candidates before doc aggregation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 in hybrid score (dense weight = 1 - alpha).",
    )
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
        help="Distance/score metric for ANN.",
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
            bm25_top=args.bm25_top,
            dense_chunks=args.dense_chunks,
            alpha=args.alpha,
            concurrency=args.concurrency,
            probes=args.probes,
            ef_search=args.ef_search,
            metric=args.metric,
            emb_path=emb_path,
            meta_path=meta_path,
        )
    )


if __name__ == "__main__":
    main()
