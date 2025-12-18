from __future__ import annotations

import argparse
import os
from typing import List, Tuple

from pymongo import MongoClient
from psycopg2.extras import execute_values

from app.config import (
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    get_connection,
)


def fetch_law_titles(uri: str, db: str, collection: str, limit: int | None = None) -> List[Tuple[str, str]]:
    client = MongoClient(uri)
    coll = client[db][collection]
    cursor = coll.find(
        {"diagram.so_hieu": {"$exists": True}, "diagram.ten": {"$exists": True}},
        {"diagram.so_hieu": 1, "diagram.ten": 1, "_id": 0},
    )
    if limit:
        cursor = cursor.limit(limit)
    pairs: List[Tuple[str, str]] = []
    for doc in cursor:
        diagram = doc.get("diagram") or {}
        so_hieu = str(diagram.get("so_hieu") or "").strip()
        title = str(diagram.get("ten") or "").strip()
        if not so_hieu or not title:
            continue
        pairs.append((so_hieu, title))
    client.close()
    return pairs


def update_titles(pairs: List[Tuple[str, str]]) -> None:
    if not pairs:
        print("No titles to update.")
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        print(f"Sample pairs: {pairs[:3]}")
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS law_titles_tmp (so_hieu text, title text);")
        cur.execute("TRUNCATE law_titles_tmp;")
        execute_values(cur, "INSERT INTO law_titles_tmp (so_hieu, title) VALUES %s", pairs, page_size=2000)

        cur.execute("SELECT COUNT(*) FROM law_titles_tmp;")
        tmp_rows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT so_hieu) FROM law_titles_tmp;")
        tmp_distinct = cur.fetchone()[0]
        print(f"Temp rows inserted: {tmp_rows} (distinct law_id: {tmp_distinct})")

        cur.execute(
            """
            SELECT COUNT(*) FROM laws l
            JOIN law_titles_tmp t ON l.law_id = t.so_hieu;
            """
        )
        match_rows = cur.fetchone()[0]
        cur.execute(
            """
            SELECT COUNT(*) FROM laws l
            JOIN law_titles_tmp t ON l.law_id = t.so_hieu
            WHERE l.title IS DISTINCT FROM t.title;
            """
        )
        diff_rows = cur.fetchone()[0]
        print(f"Matches in laws: {match_rows}; with differing title: {diff_rows}")

        cur.execute(
            """
            UPDATE laws AS l
            SET title = t.title
            FROM law_titles_tmp t
            WHERE l.law_id = t.so_hieu
              AND (l.title IS NULL OR l.title <> t.title);
            """
        )
        conn.commit()
        print(f"Updated titles for {cur.rowcount} laws.")
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync law titles from MongoDB to Postgres laws.title")
    parser.add_argument("--mongo-uri", default=MONGODB_URI, help="MongoDB connection URI")
    parser.add_argument("--mongo-db", default=MONGODB_DB, help="MongoDB database name")
    parser.add_argument("--mongo-col", default=MONGODB_COLLECTION, help="MongoDB collection name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to read from Mongo")
    args = parser.parse_args()

    if not args.mongo_uri:
        raise ValueError("Missing Mongo URI (set MONGODB_URI/MONGO_URI or pass --mongo-uri)")

    pairs = fetch_law_titles(args.mongo_uri, args.mongo_db, args.mongo_col, args.limit)
    print(f"Fetched {len(pairs)} law titles from Mongo.")
    update_titles(pairs)


if __name__ == "__main__":
    main()
