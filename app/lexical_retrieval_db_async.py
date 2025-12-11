from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import asyncpg
import numpy as np
from underthesea import word_tokenize

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, STOPWORDS_PATH
from app.data_loader import load_train_data


def load_stopwords(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        stops = {line.strip().lower() for line in f if line.strip()}
    print(f"Loaded {len(stops)} stopwords from {path}")
    return stops


def tokenize_question(text: str, stopwords: Set[str] | None = None) -> List[str]:
    """
    Tokenize a question using underthesea.
    If stopwords is None -> keep ALL tokens (including 'stopwords').
    """
    tok_str = word_tokenize(text or "", format="text")
    tokens = [t for t in tok_str.split() if t]
    if stopwords is None:
        return tokens
    return [t for t in tokens if t.lower() not in stopwords]


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


async def bm25_query(
    pool: asyncpg.pool.Pool,
    query_terms: Sequence[str],
    top_k: int,
) -> List[int]:
    """
    BM25 using PRECOMPUTED stats from the DB:

      - article_term_freq(article_id, token, tf)
      - article_stats(article_id, doc_len)
      - term_stats(token, df, idf)       -- idf already encoded
      - collection_stats(id=1, n_docs, avg_doc_len)

    Assumes article_term_freq was built from `articles.token`
    (i.e., tokens WITH stopwords).
    """
    if not query_terms:
        return []

    sql = """
    WITH params AS (
        SELECT
            $1::text[] AS tokens,
            1.2::float  AS k1,
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
    SELECT doc_id
    FROM scored
    ORDER BY score DESC
    LIMIT $2;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, list(query_terms), top_k)
    return [int(r["doc_id"]) for r in rows]


async def evaluate_variant(
    name: str,
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    top_k: int,
    concurrency: int,
) -> float:
    """
    Evaluate macro F2 for a BM25 variant.
    Uses full tokens (with stopwords) for both indexing and queries.
    """
    sem = asyncio.Semaphore(concurrency)

    async def worker(q: Dict) -> float:
        async with sem:
            # IMPORTANT: keep stopwords -> match `articles.token`
            q_tokens = tokenize_question(q["text"], stopwords=None)
            preds = await bm25_query(
                pool=pool,
                query_terms=q_tokens,
                top_k=top_k,
            )
            f2 = fbeta_score(q["gold_doc_ids"], preds, beta=2.0)
            prec, rec = precision_recall(q["gold_doc_ids"], preds)
            return f2, prec, rec

    tasks = [asyncio.create_task(worker(q)) for q in questions]
    results = await asyncio.gather(*tasks)
    if not results:
        print(f"{name}: no questions to evaluate.")
        return 0.0

    f2_scores, prec_scores, rec_scores = zip(*results)
    macro_f2 = float(np.mean(f2_scores))
    macro_prec = float(np.mean(prec_scores))
    macro_rec = float(np.mean(rec_scores))

    print(
        f"{name}: macro F2 @ {top_k} = {macro_f2:.4f} | "
        f"macro Precision = {macro_prec:.4f} | macro Recall = {macro_rec:.4f} "
        f"over {len(results)} questions"
    )
    return macro_f2


async def main_async(top_k: int, limit: int | None, concurrency: int):
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
        max_size=concurrency,
    )

    try:
        await evaluate_variant(
            name="BM25 (precomputed tf/df, FULL tokens with stopwords)",
            pool=pool,
            questions=questions,
            top_k=top_k,
            concurrency=concurrency,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(
        description="Async database-side BM25 retrieval and macro F2 evaluation (precomputed stats)."
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K documents to evaluate.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of questions for a quick check."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent DB queries (pool size).",
    )
    args = parser.parse_args()

    asyncio.run(main_async(top_k=args.top_k, limit=args.limit, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
