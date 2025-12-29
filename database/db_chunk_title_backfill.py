from __future__ import annotations

import argparse

from app.config import get_connection


def backfill_chunk_titles(overwrite_existing: bool) -> None:
    """
    Fill article_chunks.chunk_title only from articles.title (no fallbacks).
    - Adds chunk_title column if missing.
    - Defaults to only touching rows where chunk_title is NULL/empty unless --overwrite-existing is set.
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS title TEXT;")
        cur.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS chunk_title TEXT;")

        scope_filter = "" if overwrite_existing else "WHERE ac.chunk_title IS NULL OR ac.chunk_title = ''"

        cur.execute(
            f"""
            WITH base AS (
                SELECT
                    ac.id,
                    ac.chunk_index,
                    COUNT(*) OVER (PARTITION BY ac.article_fk) AS total_chunks,
                    NULLIF(trim(a.title), '') AS base_title
                FROM article_chunks ac
                JOIN articles a ON a.id = ac.article_fk
                {scope_filter}
            )
            UPDATE article_chunks ac
            SET chunk_title = CASE
                WHEN b.base_title IS NULL THEN NULL
                WHEN b.total_chunks > 1 THEN b.base_title || ' (chunk ' || b.chunk_index || '/' || b.total_chunks || ')'
                ELSE b.base_title
            END
            FROM base b
            WHERE ac.id = b.id;
            """
        )
        conn.commit()
        print(f"Updated chunk_title for {cur.rowcount} rows.")
    except Exception as e:
        conn.rollback()
        print("ERROR while backfilling chunk_title:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("DB connection closed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill article_chunks.chunk_title from articles.title (fallbacks to doc_id/law_id)."
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Recompute chunk_title even for rows that already have a value.",
    )
    args = parser.parse_args()
    backfill_chunk_titles(overwrite_existing=args.overwrite_existing)


if __name__ == "__main__":
    main()
