#!/usr/bin/env python3
"""
Import laws + articles from the Mongo `processed_documents` collection into Postgres,
then chunk and rebuild chunk-level BM25 stats for the newly processed rows.

The extraction logic is shared with scripts/extract_articles.py.

Example:
  python database/db_law_mongodb.py --mongo-ids 693300a913b0444ae980c268
  python database/db_law_mongodb.py --law-ids 2000 --replace-chunks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from bson import ObjectId
from pymongo import MongoClient
from psycopg2.extras import execute_values
from underthesea import word_tokenize

# Ensure project root is on sys.path so we can import sibling modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    get_connection,
)
from database import db_chunk_article as chunk_mod  # noqa: E402
from database.db_chunk_bm25_stats import rebuild_chunk_bm25  # noqa: E402
from scripts.extract_articles import (  # noqa: E402
    extract_articles_from_document,
    get_article_full_content,
    get_mongodb_client,
)

# ----------------------------
# Helpers
# ----------------------------


def _norm_law_id(val: str) -> str:
    """Normalize a law identifier for comparisons (lowercase + strip whitespace)."""
    return "".join((val or "").lower().split())


def tokenize(text: str) -> List[str]:
    """
    Tokenize while preserving stopwords (only `token` is stored; no stopword filtering).
    """
    tok_str = word_tokenize(text or "", format="text")
    return [t.lower() for t in tok_str.split() if t]


def parse_object_ids(raw_ids: Iterable[str]) -> List[ObjectId]:
    out: List[ObjectId] = []
    for raw in raw_ids:
        try:
            out.append(ObjectId(str(raw)))
        except Exception as exc:
            raise ValueError(f"Invalid ObjectId: {raw}") from exc
    return out


def derive_law_info(doc: Dict[str, Any], result) -> tuple[str, str, Optional[str]]:
    """
    law_id comes from document_metadata.number (preferred) or diagram.so_hieu.
    law_title comes from document_metadata.title (preferred) or diagram.ten.
    """
    diagram = doc.get("diagram") or {}
    meta = result.document_metadata

    law_id = str(meta.get("number") or diagram.get("so_hieu") or "").strip()
    if not law_id:
        raise ValueError("Missing law identifier (document_metadata.number or diagram.so_hieu)")

    law_title = str(meta.get("title") or diagram.get("ten") or result.document_title or law_id).strip()
    law_type = meta.get("type") or diagram.get("loai_van_ban") or None
    return law_id, law_title, law_type


def derive_article_title(article) -> Optional[str]:
    """
    Article title = raw article.content (not full_content), to keep alignment with extractor output.
    """
    if article.content:
        return str(article.content).strip()
    return None


def make_doc_id_allocator(cur, start_at: Optional[int]) -> Optional[Callable[[], int]]:
    if start_at is None:
        cur.execute("SELECT COALESCE(MAX(doc_id), 0) FROM articles;")
        start_at = cur.fetchone()[0] or 0

    state = {"current": int(start_at)}

    def _next() -> int:
        state["current"] += 1
        return state["current"]

    return _next


def fetch_documents(
    client: MongoClient,
    mongo_ids: List[ObjectId] | None,
    law_ids: List[str] | None,
    limit: int | None,
) -> List[dict]:
    query: Dict[str, Any] = {}
    if mongo_ids:
        query["_id"] = {"$in": mongo_ids}
    if law_ids:
        query["diagram.so_hieu"] = {"$in": [str(x) for x in law_ids]}

    coll = client[MONGODB_DB][MONGODB_COLLECTION]
    cursor = coll.find(query)
    if limit:
        cursor = cursor.limit(limit)

    docs = list(cursor)
    if not docs:
        raise ValueError("No Mongo documents matched the query filters.")
    return docs


def load_existing_law_norms(cur) -> set[str]:
    cur.execute("SELECT law_id FROM laws;")
    norms = {_norm_law_id(row[0]) for row in cur.fetchall() if row and row[0]}
    return norms


def upsert_law(cur, law_id: str, title: str, source: str, law_type: Optional[str]) -> int:
    cur.execute("ALTER TABLE laws ADD COLUMN IF NOT EXISTS doc_type TEXT;")
    cur.execute(
        """
        INSERT INTO laws (law_id, title, source, doc_type)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (law_id) DO UPDATE
        SET title = COALESCE(NULLIF(EXCLUDED.title, ''), laws.title),
            source = EXCLUDED.source,
            doc_type = COALESCE(NULLIF(EXCLUDED.doc_type, ''), laws.doc_type)
        RETURNING id;
        """,
        (law_id, title, source, law_type),
    )
    row = cur.fetchone()
    return int(row[0])


def upsert_articles(
    cur,
    law_pk: int,
    law_id: str,
    result,
    source: str,
    next_doc_id: Optional[Callable[[], int]],
) -> List[Dict[str, Any]]:
    payloads: List[tuple] = []
    meta: List[Dict[str, Any]] = []
    seen_article_ids: set[str] = set()

    for article in result.articles:
        if not article.id:
            continue
        if article.id in seen_article_ids:
            continue  # avoid duplicate (law_id, article_id) in the same batch
        seen_article_ids.add(article.id)

        full_text = (get_article_full_content(article) or "").strip()
        if not full_text:
            continue

        tokens = tokenize(full_text)
        doc_id = next_doc_id() if next_doc_id else None
        title = derive_article_title(article)
        article_idx = (
            str(article.article_number) if article.article_number is not None else None
        )

        payloads.append(
            (
                law_pk,
                law_id,
                article.id,
                article_idx,
                title,
                full_text,
                tokens,
                None,
                source,
                doc_id,
            )
        )
        meta.append({"article_id": article.id, "title": title, "text": full_text})

    if not payloads:
        return []

    rows = execute_values(
        cur,
        """
        INSERT INTO articles (
            law_fk, law_id, article_id, article_idx, title, text, token,
            token_no_stopword, source, doc_id
        )
        VALUES %s
        ON CONFLICT (law_id, article_id) DO UPDATE
        SET law_fk            = EXCLUDED.law_fk,
            article_idx       = COALESCE(EXCLUDED.article_idx, articles.article_idx),
            title             = COALESCE(NULLIF(EXCLUDED.title, ''), articles.title),
            text              = EXCLUDED.text,
            token             = EXCLUDED.token,
            token_no_stopword = EXCLUDED.token_no_stopword,
            source            = EXCLUDED.source,
            doc_id            = COALESCE(articles.doc_id, EXCLUDED.doc_id)
        RETURNING id, article_id, doc_id, title;
        """,
        payloads,
        fetch=True,
        page_size=200,
    )

    inserted: List[Dict[str, Any]] = []
    for info, row in zip(meta, rows):
        inserted.append(
            {
                "id": int(row[0]),
                "article_id": str(row[1]),
                "doc_id": row[2],
                "title": row[3] or info["title"],
                "text": info["text"],
            }
        )
    return inserted


def ensure_chunk_schema(cur) -> None:
    cur.execute(chunk_mod.CHUNK_TABLE_SQL)
    cur.execute(
        "ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS chunk_title TEXT;"
    )
    cur.execute(
        "ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_with_title_bge_m3 vector(1024);"
    )
    cur.execute(
        "ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector(1024);"
    )
    cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS title TEXT;")


def upsert_chunks(
    cur,
    articles: List[Dict[str, Any]],
    replace: bool,
    batch_size: int = 3000,
) -> int:
    if not articles:
        return 0

    ensure_chunk_schema(cur)
    # Initialize worker without stopword removal; token_no_stopword will be left NULL.
    chunk_mod._init_worker([])

    article_ids = [a["id"] for a in articles]
    if replace:
        cur.execute(
            "DELETE FROM article_chunks WHERE article_fk = ANY(%s);", (article_ids,)
        )

    sql = """
    INSERT INTO article_chunks (
        article_fk, doc_id, chunk_title, chunk_index, char_start, char_end, text,
        token, token_no_stopword
    )
    VALUES %s
    ON CONFLICT (article_fk, chunk_index) DO UPDATE
    SET doc_id            = EXCLUDED.doc_id,
        chunk_title       = EXCLUDED.chunk_title,
        char_start        = EXCLUDED.char_start,
        char_end          = EXCLUDED.char_end,
        text              = EXCLUDED.text,
        token             = EXCLUDED.token,
        token_no_stopword = EXCLUDED.token_no_stopword;
    """

    buffer: List[tuple] = []
    inserted_rows = 0
    for art in articles:
        rows = chunk_mod.process_article(
            art["id"], art["doc_id"], art["text"], art["title"]
        )
        if not rows:
            continue
        # Force token_no_stopword to NULL to keep only tokens.
        for (
            article_fk,
            doc_id,
            chunk_title,
            chunk_index,
            char_start,
            char_end,
            text,
            tokens,
            _tokens_no_stop,
        ) in rows:
            buffer.append(
                (
                    article_fk,
                    doc_id,
                    chunk_title,
                    chunk_index,
                    char_start,
                    char_end,
                    text,
                    tokens,
                    None,
                )
            )

        if len(buffer) >= batch_size:
            execute_values(cur, sql, buffer, page_size=2000)
            inserted_rows += len(buffer)
            buffer.clear()

    if buffer:
        execute_values(cur, sql, buffer, page_size=2000)
        inserted_rows += len(buffer)

    return inserted_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import tvpl laws/articles from Mongo into Postgres and chunk them."
    )
    parser.add_argument(
        "--mongo-ids",
        nargs="+",
        help="Mongo _id values (processed_documents) to ingest.",
    )
    parser.add_argument(
        "--law-ids",
        nargs="+",
        help="Filter by diagram.so_hieu values (law identifiers).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of Mongo documents to load.",
    )
    parser.add_argument("--mongo-uri", default=MONGODB_URI, help="Mongo connection URI.")
    parser.add_argument("--source", default="tvpl", help="Source tag stored in DB.")
    parser.add_argument(
        "--start-doc-id",
        type=int,
        default=None,
        help="Manual starting doc_id (default: current MAX(doc_id)).",
    )
    parser.add_argument(
        "--no-auto-doc-id",
        action="store_true",
        help="Leave doc_id NULL (chunks/BM25 will skip those rows).",
    )
    parser.add_argument(
        "--no-chunk", action="store_true", help="Skip chunking the imported articles."
    )
    parser.add_argument(
        "--replace-chunks",
        action="store_true",
        help="Delete existing chunks for targeted articles before inserting new ones.",
    )
    parser.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Skip rebuilding chunk BM25 stats after chunking.",
    )
    parser.add_argument(
        "--bm25-no-truncate",
        action="store_true",
        help="When rebuilding BM25, avoid truncating helper tables (incremental update).",
    )
    parser.add_argument(
        "--auto-new-laws",
        type=int,
        default=None,
        help="Stream Mongo docs until N new laws with articles (unique by diagram.so_hieu) are ingested.",
    )
    parser.add_argument(
        "--doc-types",
        nargs="+",
        default=None,
        help='Only ingest laws whose type matches these values (case-insensitive), e.g. "Luật" "Nghị định".',
    )
    args = parser.parse_args()

    if not args.mongo_ids and not args.law_ids and not args.limit and not args.auto_new_laws:
        raise SystemExit(
            "Specify one of --mongo-ids, --law-ids, --limit, or --auto-new-laws to avoid scanning the full collection."
        )

    # Mongo connection
    mongo_client = (
        MongoClient(args.mongo_uri)
        if args.mongo_uri
        else get_mongodb_client()
    )

    conn = None
    cur = None
    docs = None
    docs_cursor = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        doc_types_norm = [dt.lower().strip() for dt in (args.doc_types or []) if dt] or None

        existing_norms = load_existing_law_norms(cur)
        print(f"Loaded {len(existing_norms)} existing law_ids from Postgres.")

        if args.auto_new_laws:
            docs_cursor = (
                mongo_client[MONGODB_DB][MONGODB_COLLECTION]
                .find({"diagram.so_hieu": {"$exists": True}}, no_cursor_timeout=True)
                .batch_size(250)
            )
            print(
                f"Streaming Mongo docs until {args.auto_new_laws} new laws with articles are ingested..."
            )
        else:
            mongo_ids = parse_object_ids(args.mongo_ids) if args.mongo_ids else None
            docs = fetch_documents(mongo_client, mongo_ids, args.law_ids, args.limit)

        next_doc_id_fn = None
        if not args.no_auto_doc_id:
            cur.execute("SELECT COALESCE(MAX(doc_id), 0) FROM articles;")
            current_max_doc_id = cur.fetchone()[0] or 0
            if args.start_doc_id is not None and args.start_doc_id < current_max_doc_id:
                raise ValueError(
                    f"--start-doc-id ({args.start_doc_id}) is below current MAX(doc_id) {current_max_doc_id}"
                )
            start_at = args.start_doc_id if args.start_doc_id is not None else current_max_doc_id
            next_doc_id_fn = make_doc_id_allocator(cur, start_at)
            print(f"doc_id allocator starting at {start_at}")

        total_articles = 0
        chunk_targets: List[Dict[str, Any]] = []
        skipped_existing = 0
        skipped_invalid = 0
        skipped_no_articles = 0
        skipped_doc_type = 0
        seen_norms_run: set[str] = set()
        ingested_laws_with_articles = 0

        iterable = docs if docs is not None else docs_cursor  # type: ignore[arg-type]

        for doc in iterable:
            try:
                # Ensure extractor-friendly defaults for missing keys
                safe_doc = dict(doc)
                if not safe_doc.get("table_of_content"):
                    safe_doc["table_of_content"] = {}
                if "diagram" not in safe_doc or safe_doc["diagram"] is None:
                    safe_doc["diagram"] = {}

                result = extract_articles_from_document(safe_doc)
            except Exception as exc:
                skipped_invalid += 1
                print(f"Skipping doc due to extraction error: {exc!r}")
                continue
            law_id, law_title, law_type = derive_law_info(doc, result)
            law_type_norm = law_type.lower().strip() if isinstance(law_type, str) else ""

            if doc_types_norm and law_type_norm not in doc_types_norm:
                skipped_doc_type += 1
                continue

            norm = _norm_law_id(law_id)
            # Allow re-ingesting laws even if they already exist in Postgres (e.g., different source).
            # Only skip duplicates within the same run to avoid double-processing the same doc.
            if norm in seen_norms_run:
                skipped_existing += 1
                continue

            law_pk = upsert_law(cur, law_id, law_title, args.source, law_type)

            inserted = upsert_articles(
                cur,
                law_pk,
                law_id,
                result,
                args.source,
                next_doc_id_fn,
            )
            if not inserted:
                skipped_no_articles += 1
                # Clean up the newly inserted law row so laws only exist when articles do.
                cur.execute("DELETE FROM laws WHERE id = %s;", (law_pk,))
                continue
            total_articles += len(inserted)
            chunk_targets.extend(inserted)
            seen_norms_run.add(norm)
            ingested_laws_with_articles += 1
            print(
                f"Ingested law_id={law_id} (articles processed: {len(inserted)})"
            )
            if args.auto_new_laws and ingested_laws_with_articles >= args.auto_new_laws:
                break

        conn.commit()
        print(
            f"✅ Upserted {total_articles} articles across {len(seen_norms_run)} laws "
            f"(skipped existing: {skipped_existing}, skipped invalid: {skipped_invalid}, "
            f"skipped with no articles: {skipped_no_articles}, skipped by doc_type: {skipped_doc_type})."
        )
        if args.auto_new_laws and ingested_laws_with_articles < args.auto_new_laws:
            print(
                f"⚠️ Requested {args.auto_new_laws} laws with articles, "
                f"but only ingested {ingested_laws_with_articles} before cursor was exhausted."
            )

        if args.no_chunk or not chunk_targets:
            print("Skipping chunking step.")
            return

        chunk_rows = upsert_chunks(
            cur,
            chunk_targets,
            replace=args.replace_chunks,
        )
        conn.commit()
        print(f"Inserted/updated {chunk_rows} article_chunks rows.")

        if not args.skip_bm25:
            rebuild_chunk_bm25(truncate=not args.bm25_no_truncate)
        else:
            print("Skipped chunk BM25 rebuild.")

    finally:
        try:
            if docs_cursor is not None:
                docs_cursor.close()
            if cur:
                cur.close()
            if conn:
                conn.close()
        except Exception:
            pass
        mongo_client.close()


if __name__ == "__main__":
    main()
