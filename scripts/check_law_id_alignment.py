from __future__ import annotations

import argparse
from typing import Dict, List

from pymongo import MongoClient

from app.config import (
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    get_connection,
)
from scripts.sync_article_idx_title_from_mongo import _extract_articles, _norm_law_id


def _load_pg_laws() -> Dict[str, dict]:
    """
    Return norm_law_id -> info from Postgres (articles table).
    Aggregates by normalized law_id to make comparisons with Mongo easier.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT law_id,
               COUNT(*) AS article_rows,
               COUNT(title) FILTER (WHERE title IS NOT NULL AND title <> '') AS titled_rows
        FROM articles
        GROUP BY law_id;
        """
    )
    mapping: Dict[str, dict] = {}
    for law_id, article_rows, titled_rows in cur.fetchall():
        norm = _norm_law_id(law_id)
        if not norm:
            continue
        info = mapping.setdefault(
            norm,
            {"law_ids": set(), "article_rows": 0, "titled_rows": 0},
        )
        info["law_ids"].add(law_id)
        info["article_rows"] += int(article_rows)
        info["titled_rows"] += int(titled_rows)
    cur.close()
    conn.close()
    return mapping


def _load_mongo_laws(uri: str, db: str, collection: str, limit: int | None = None) -> Dict[str, dict]:
    """
    Return norm_law_id -> info from Mongo (diagram.so_hieu + TOC length).
    Keeps the version with the largest TOC for duplicate so_hieu values.
    """
    client = MongoClient(uri)
    coll = client[db][collection]
    cursor = coll.find(
        {"diagram.so_hieu": {"$exists": True}},
        {"diagram.so_hieu": 1, "diagram.loai_van_ban": 1, "table_of_content": 1, "_id": 0},
    )
    if limit:
        cursor = cursor.limit(limit)

    mapping: Dict[str, dict] = {}
    for doc in cursor:
        law_id, articles = _extract_articles(doc)
        if not law_id:
            continue
        norm = _norm_law_id(law_id)
        if not norm:
            continue

        toc_count = len(articles)
        if toc_count == 0:
            continue

        prev = mapping.get(norm)
        if prev and prev["toc_count"] >= toc_count:
            continue
        mapping[norm] = {
            "law_id": law_id,
            "loai_van_ban": (doc.get("diagram") or {}).get("loai_van_ban"),
            "toc_count": toc_count,
        }

    client.close()
    return mapping


def _print_examples(title: str, rows: List[str], data: Dict[str, dict], limit: int) -> None:
    if not rows:
        return
    print(f"\n{title} ({len(rows)}) - showing up to {limit}:")
    for norm in rows[:limit]:
        info = data[norm]
        sample_law_id = next(iter(info.get("law_ids") or [info.get("law_id") or ""]))
        extra = []
        if "article_rows" in info:
            extra.append(f"pg_articles={info['article_rows']}")
            extra.append(f"titled={info['titled_rows']}")
        if "toc_count" in info:
            extra.append(f"mongo_toc={info['toc_count']}")
            loai = info.get("loai_van_ban")
            if loai:
                extra.append(f"loai={loai}")
        suffix = f"; {' | '.join(extra)}" if extra else ""
        print(f"  {sample_law_id} (norm={norm}{suffix})")


def _print_mismatch_examples(rows: List[str], pg_map: Dict[str, dict], mongo_map: Dict[str, dict], limit: int) -> None:
    if not rows:
        return
    print(f"\nArticle count/title mismatches ({len(rows)}) - showing up to {limit}:")
    for norm in rows[:limit]:
        pg_info = pg_map[norm]
        mongo_info = mongo_map[norm]
        pg_law = next(iter(pg_info["law_ids"]))
        mongo_law = mongo_info.get("law_id") or ""
        print(
            f"  norm={norm} | pg_law_id={pg_law} articles={pg_info['article_rows']} (titled={pg_info['titled_rows']})"
            f" | mongo_so_hieu={mongo_law} toc_entries={mongo_info['toc_count']} loai={mongo_info.get('loai_van_ban')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit overlap between Postgres law_id values and Mongo diagram.so_hieu"
    )
    parser.add_argument("--mongo-uri", default=MONGODB_URI, help="MongoDB connection URI")
    parser.add_argument("--mongo-db", default=MONGODB_DB, help="MongoDB database name")
    parser.add_argument("--mongo-col", default=MONGODB_COLLECTION, help="MongoDB collection name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of Mongo docs to scan (for quick checks)")
    parser.add_argument(
        "--show",
        type=int,
        default=20,
        help="Number of sample rows to show for each mismatch bucket",
    )
    args = parser.parse_args()

    if not args.mongo_uri:
        raise ValueError("Missing Mongo URI (set MONGODB_URI/MONGO_URI or pass --mongo-uri)")

    pg_map = _load_pg_laws()
    mongo_map = _load_mongo_laws(args.mongo_uri, args.mongo_db, args.mongo_col, args.limit)

    pg_norms = set(pg_map)
    mongo_norms = set(mongo_map)
    overlap = pg_norms & mongo_norms

    missing_in_mongo = sorted(pg_norms - mongo_norms)
    missing_in_pg = sorted(mongo_norms - pg_norms)

    # Laws present in both sides but with mismatched article counts or missing titles.
    count_mismatches: List[str] = []
    for norm in sorted(overlap):
        pg_info = pg_map[norm]
        mongo_info = mongo_map[norm]
        if pg_info["article_rows"] != mongo_info["toc_count"] or pg_info["titled_rows"] < pg_info["article_rows"]:
            count_mismatches.append(norm)

    print(f"Postgres laws (normalized): {len(pg_norms)}")
    print(f"Mongo so_hieu (normalized): {len(mongo_norms)}")
    print(f"Overlap: {len(overlap)}")
    print(f"Missing in Mongo: {len(missing_in_mongo)}")
    print(f"Missing in Postgres: {len(missing_in_pg)}")
    print(f"Overlap with article-count or title coverage mismatch: {len(count_mismatches)}")

    _print_examples("Missing in Mongo", missing_in_mongo, pg_map, args.show)
    _print_examples("Missing in Postgres", missing_in_pg, mongo_map, args.show)
    _print_mismatch_examples(count_mismatches, pg_map, mongo_map, args.show)


if __name__ == "__main__":
    main()
