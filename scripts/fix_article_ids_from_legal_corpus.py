from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

import asyncpg

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, LAW_PATH


def _load_corpus(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_docid_to_posidx(corpus: List[dict]) -> Dict[int, str]:
    """
    Build mapping: doc_id (aid) -> positional index within its law (1..N).
    No parsing of article numbers or text.
    """
    mapping: Dict[int, str] = {}

    for law in corpus:
        articles = law.get("content") or []
        for pos, art in enumerate(articles, start=1):
            aid = art.get("aid")
            if aid is None:
                continue
            try:
                doc_id = int(aid)
            except (TypeError, ValueError):
                continue

            mapping[doc_id] = str(pos)

    return mapping


async def update_article_idx(mapping: Dict[int, str], concurrency: int) -> None:
    if not mapping:
        print("No article idx to update.")
        return

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=max(2, concurrency),
    )

    # Ensure column exists
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS article_idx text;")

    sem = asyncio.Semaphore(concurrency)
    updated = 0
    skipped = 0

    async def worker(doc_id: int, idx: str) -> None:
        nonlocal updated, skipped
        async with sem, pool.acquire() as conn:
            res = await conn.execute(
                """
                UPDATE articles
                SET article_idx = $1
                WHERE doc_id = $2
                  AND (article_idx IS DISTINCT FROM $1);
                """,
                idx,
                doc_id,
            )
            # res looks like "UPDATE 0" or "UPDATE 1"
            if res.endswith("0"):
                skipped += 1
            else:
                updated += 1

    tasks = [asyncio.create_task(worker(doc_id, idx)) for doc_id, idx in mapping.items()]
    await asyncio.gather(*tasks)

    await pool.close()
    print(f"Doc_ids processed: {len(mapping)}")
    print(f"Updated rows: {updated}")
    print(f"Skipped (not found or already same): {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set articles.article_idx as positional index (1..N) per law")
    parser.add_argument("--corpus-path", type=Path, default=LAW_PATH, help="Path to legal_corpus.json")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    corpus = _load_corpus(args.corpus_path)
    mapping = build_docid_to_posidx(corpus)
    print(f"Built positional mapping for {len(mapping)} doc_ids from {args.corpus_path}")

    asyncio.run(update_article_idx(mapping, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
