from __future__ import annotations

import argparse
import asyncio
import re
from typing import Dict, List, Tuple

import asyncpg
from pymongo import MongoClient

from app.config import (
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME,
)


def _norm_law_id(val: str) -> str:
    """Normalize law_id for matching: lowercase, remove all whitespace."""
    return "".join(val.lower().split())


def _collect_entries(nodes: List[dict], pattern: re.Pattern[str]) -> List[Tuple[str, str]]:
    """DFS through TOC nodes, collecting (article_idx, title) for keys matching pattern."""
    out: List[Tuple[str, str]] = []
    for node in nodes:
        key = str(node.get("key") or "")
        label = str(node.get("label") or "").strip()

        m = pattern.match(key.lower())
        if m:
            art_idx_raw = m.group(1)
            art_idx = art_idx_raw.lstrip("0") or art_idx_raw
            if art_idx and label:
                out.append((art_idx, label))

        children = node.get("children") or []
        if children:
            out.extend(_collect_entries(children, pattern))
    return out


def _extract_articles(doc: dict) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extract per-law article indices and titles from Mongo TOC.
    law_id comes from diagram.so_hieu.
    Prefer dieu_* entries; if none are found, fall back to muc_* (some laws only use muc_*).
    """
    diagram = doc.get("diagram") or {}
    law_id = str(diagram.get("so_hieu") or "").strip()
    toc_items = (doc.get("table_of_content") or {}).get("items") or []
    if not toc_items:
        return law_id, []

    dieu_pattern = re.compile(r"^#?dieu[\s_-]*([0-9a-zA-Z]+)")
    muc_pattern = re.compile(r"^#?muc[\s_-]*([0-9a-zA-Z]+)")

    articles: List[Tuple[str, str]] = []
    # Try dieu_* first (always).
    for item in toc_items:
        # Run DFS starting from the item and its immediate children to catch dieu_* at either level.
        roots = [item]
        roots.extend(item.get("children") or [])
        articles.extend(_collect_entries(roots, dieu_pattern))

    # If no dieu_* found, fall back to muc_* (often under items.children).
    if not articles:
        muc_roots: List[dict] = []
        for item in toc_items:
            muc_roots.append(item)
            muc_roots.extend(item.get("children") or [])
        articles.extend(_collect_entries(muc_roots, muc_pattern))

    return law_id, articles


def _norm_loai(loai: str) -> str:
    return "".join(loai.lower().split())


def fetch_from_mongo(
    uri: str,
    db: str,
    collection: str,
    limit: int | None = None,
    target_norm_law_ids: set[str] | None = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build mapping norm_law_id -> [(article_idx, title)...].
    Chooses the best TOC per so_hieu using a priority list (case-insensitive; supports TT/TTLN aliases),
    breaking ties by larger article count.
    If target_norm_law_ids is provided, only keeps so_hieu that exactly match that normalized set.
    """
    client = MongoClient(uri)
    coll = client[db][collection]

    priority_list = [
        "Luật",
        "Nghị định",
        "Thông Tư",
        "Thông tư liên tịch",
        "Quyết định",
        "Văn bản hợp nhất",
        "Quy chế",
        "Nghị định thư",
        "Hiến pháp",
        "Pháp lệnh",
        "Thông báo",
        "Chỉ thị",
        "Hiệp định",
        "Điều ước quốc tế",
        "Kế hoạch",
        "Hướng dẫn",
        "Văn bản khác",
        "Thông tri",
        "WTO_Cam kết VN",
        "Sắc luật",
        "Thoả thuận",
        "Sắc lệnh",
        "Điều lệ",
    ]
    priority_map = {_norm_loai(loai): idx for idx, loai in enumerate(priority_list)}

    cursor = coll.find(
        {
            "diagram.so_hieu": {"$exists": True},
            "diagram.loai_van_ban": {"$exists": True},
            "table_of_content.items": {"$exists": True},
        },
        {
            "diagram.so_hieu": 1,
            "diagram.loai_van_ban": 1,
            "table_of_content": 1,
            "_id": 0,
        },
    )
    if limit:
        cursor = cursor.limit(limit)

    best: Dict[str, dict] = {}

    for doc in cursor:
        law_id, articles = _extract_articles(doc)
        if not law_id or not articles:
            continue
        norm = _norm_law_id(law_id)
        if not norm:
            continue
        if target_norm_law_ids is not None and norm not in target_norm_law_ids:
            continue

        raw_loai = str((doc.get("diagram") or {}).get("loai_van_ban") or "")
        norm_loai = _norm_loai(raw_loai)
        priority = priority_map.get(norm_loai, len(priority_map) + 10)

        prev = best.get(norm)
        if prev:
            if priority > prev["priority"]:
                continue
            if priority == prev["priority"] and len(prev["articles"]) >= len(articles):
                continue

        best[norm] = {"articles": articles, "priority": priority}

    client.close()
    return {k: v["articles"] for k, v in best.items()}


async def _ensure_columns(pool: asyncpg.pool.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS article_idx text;")
        await conn.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS title text;")


async def _update_one_law(
    pool: asyncpg.pool.Pool,
    norm_law_id: str,
    mongo_articles: List[Tuple[str, str]],
) -> Tuple[int, int]:
    """
    Updates missing titles (and article_idx for those rows) by matching article_idx where possible,
    falling back to positional zip. Returns (updated_rows, skipped_flag).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, article_idx, title
            FROM articles
            WHERE regexp_replace(lower(law_id), '\s+', '', 'g') = $1
            ORDER BY
              CASE WHEN article_idx ~ '^\d+$' THEN article_idx::int END,
              id
            """,
            norm_law_id,
        )
        if not rows:
            return (0, 1)

        # Track which rows already have titles; we only update missing titles.
        has_title = {
            int(r["id"]): bool(r["title"] and str(r["title"]).strip())
            for r in rows
        }

        # Build a map by article_idx (if present), otherwise by row order fallback
        idx_map = {str(r["article_idx"]): int(r["id"]) for r in rows if r["article_idx"] is not None}
        # fallback: if no article_idx in DB, try positional zip (full order to keep alignment)
        by_position: List[int] = [int(r["id"]) for r in rows]

        updates: List[Tuple[str, str, int]] = []
        matched = 0
        max_len = min(len(by_position), len(mongo_articles))
        for pos, (art_idx, title) in zip(range(max_len), mongo_articles):
            pk = idx_map.get(str(art_idx))
            if pk is None and pos < len(by_position):
                pk = by_position[pos]
            if pk is None:
                continue
            if has_title.get(pk, False):
                # Skip rows that already have a title; requirement is to fill only NULL/empty titles.
                continue
            matched += 1
            updates.append((art_idx, title, pk))

        if not updates:
            print(f"[SKIP] law={norm_law_id} reason=no_updates")
            return (0, 1)

        await conn.executemany(
            """
            UPDATE articles
            SET article_idx = $1,
                title = $2
            WHERE id = $3
              AND (title IS NULL OR title = '')
              AND (article_idx IS DISTINCT FROM $1 OR title IS DISTINCT FROM $2);
            """,
            updates,
        )
        return (len(updates), 0)


async def sync_articles(mapping: Dict[str, List[Tuple[str, str]]], concurrency: int) -> None:
    if not mapping:
        print("No data to sync.")
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

    try:
        await _ensure_columns(pool)

        sem = asyncio.Semaphore(concurrency)
        updated_total = 0
        skipped_total = 0

        async def worker(norm_id: str, arts: List[Tuple[str, str]]):
            nonlocal updated_total, skipped_total
            async with sem:
                upd, skip = await _update_one_law(pool, norm_id, arts)
                updated_total += upd
                skipped_total += skip

        tasks = [asyncio.create_task(worker(nid, arts)) for nid, arts in mapping.items()]
        await asyncio.gather(*tasks)

        print(f"Laws processed: {len(mapping)}")
        print(f"Laws skipped (not found / count mismatch): {skipped_total}")
        print(f"Articles updated (attempted): {updated_total}")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync per-law article_idx and title from Mongo TOC (#dieu_*) into Postgres articles"
    )
    parser.add_argument("--mongo-uri", default=MONGODB_URI, help="MongoDB connection URI")
    parser.add_argument("--mongo-db", default=MONGODB_DB, help="MongoDB database name")
    parser.add_argument("--mongo-col", default=MONGODB_COLLECTION, help="MongoDB collection name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of Mongo docs to scan")
    parser.add_argument("--concurrency", type=int, default=8, help="Async concurrency for Postgres updates")
    args = parser.parse_args()

    if not args.mongo_uri:
        raise ValueError("Missing Mongo URI (set MONGODB_URI/MONGO_URI or pass --mongo-uri)")

    # Load the exact law_ids we need to update (normalized) to avoid partial matches.
    import psycopg2

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT regexp_replace(lower(law_id), '\\s+', '', 'g') FROM articles;")
    target_norms = {row[0] for row in cur.fetchall() if row[0]}
    cur.close()
    conn.close()

    mapping = fetch_from_mongo(
        args.mongo_uri, args.mongo_db, args.mongo_col, args.limit, target_norm_law_ids=target_norms
    )
    print(f"Fetched {len(mapping)} laws with article_idx/title from Mongo.")

    asyncio.run(sync_articles(mapping, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
