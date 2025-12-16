from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import asyncpg
from pgvector.asyncpg import register_vector

from app.semantic_eval_utils import tokenize_question


async def init_pgvector(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


# -------- BM25 (doc-level) --------
async def bm25_query(
    pool: asyncpg.pool.Pool,
    query_terms: Sequence[str],
    top_k: int,
    k1: float = 1.2,
    b: float = 0.75,
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
            $2::float AS k1,
            $3::float AS b
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
    LIMIT $4;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, tokens, k1, b, top_k)
    return [(int(r["doc_id"]), float(r["score"])) for r in rows]


# -------- ANN (chunk-level aggregated to doc) --------
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


def tokenize_and_bm25_terms(text: str) -> List[str]:
    return tokenize_question(text)


def min_max_norm(pairs: List[tuple[int, float]]) -> Dict[int, float]:
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
    bm25_norm = min_max_norm(bm25_pairs)
    dense_norm = min_max_norm(dense_pairs)
    doc_ids = set(bm25_norm) | set(dense_norm)

    combined: List[tuple[int, float]] = []
    for doc in doc_ids:
        b = bm25_norm.get(doc, 0.0)
        d = dense_norm.get(doc, 0.0)
        combined.append((doc, alpha * b + (1 - alpha) * d))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in combined]
