from __future__ import annotations

import asyncio
from typing import Dict, List, Sequence, Set

import asyncpg
from underthesea import word_tokenize

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.data_loader import load_train_data


def tokenize_question(text: str) -> List[str]:
    """
    Tokenize a question using underthesea.
    Keep ALL tokens (including 'stopwords') to match articles.token.
    """
    tok_str = word_tokenize(text or "", format="text")
    tokens = [t.lower() for t in tok_str.split() if t]
    return tokens


async def bm25_query(
    pool: asyncpg.pool.Pool,
    query_terms: Sequence[str],
    top_k: int,
) -> List[int]:
    """
    BM25 using PRECOMPUTED stats from the DB:

      - article_term_freq(article_id, token, tf)
      - article_stats(article_id, doc_len)
      - term_stats(token, df, idf)
      - collection_stats(id=1, n_docs, avg_doc_len)

    article_id in these tables = articles.doc_id
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


async def main_async(top_k: int = 500, max_print: int = 30):
    questions = load_train_data()
    print(f"Loaded {len(questions)} questions")

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=10,
    )

    zero_recall_count = 0
    printed = 0

    try:
        for idx, q in enumerate(questions):
            gold: Set[int] = set(q["gold_doc_ids"])
            if not gold:
                # Skip questions with no labelled gold docs (if any)
                continue

            q_tokens = tokenize_question(q["text"])
            preds = await bm25_query(pool, q_tokens, top_k=top_k)
            pred_set = set(preds)

            intersection = gold & pred_set
            if not intersection:
                zero_recall_count += 1
                if printed < max_print:
                    printed += 1
                    print("============================================")
                    print(f"Question index: {idx}")
                    print(f"Question ID   : {q.get('question_id')}")
                    print("Question text :")
                    print(q["text"])
                    print("GOLD doc_ids  :", sorted(gold))
                    print(f"Top-{top_k} BM25 doc_ids (first 20):", preds[:20])

        print("============================================")
        print(f"Questions with recall = 0.0 (top-{top_k}): {zero_recall_count}")
        print(f"Out of {len(questions)} total questions.")
    finally:
        await pool.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
