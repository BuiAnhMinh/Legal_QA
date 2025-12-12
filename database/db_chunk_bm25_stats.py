from __future__ import annotations

import argparse

from app.config import get_connection


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunk_stats (
    chunk_id    BIGINT PRIMARY KEY,
    doc_id      INT,
    doc_len     INT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_term_freq (
    chunk_id    INT NOT NULL,
    token       TEXT NOT NULL,
    tf          INT NOT NULL,
    PRIMARY KEY (chunk_id, token)
);

CREATE TABLE IF NOT EXISTS chunk_term_stats (
    token TEXT PRIMARY KEY,
    df    INT NOT NULL,
    idf   DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk_collection_stats (
    id          INT PRIMARY KEY CHECK (id = 1),
    n_docs      INT NOT NULL,
    avg_doc_len DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_term_token
    ON chunk_term_freq (token, chunk_id);
"""


def rebuild_chunk_bm25(truncate: bool = True) -> None:
    """
    Precompute BM25 statistics for chunk-level retrieval:
      - chunk_stats:      per-chunk length + doc_id
      - chunk_term_freq:  per (chunk, token) tf
      - chunk_collection_stats: n_docs + avg_doc_len
      - chunk_term_stats: df + idf
    """
    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(SCHEMA_SQL)
        if truncate:
            cur.execute(
                """
                TRUNCATE TABLE
                    chunk_stats,
                    chunk_term_freq,
                    chunk_term_stats,
                    chunk_collection_stats;
                """
            )
            print("[chunk-bm25] Truncated chunk BM25 tables.")

        # 1) Per-chunk length (only chunks with a doc_id)
        cur.execute(
            """
            INSERT INTO chunk_stats (chunk_id, doc_id, doc_len)
            SELECT
                ac.id,
                ac.doc_id,
                cardinality(COALESCE(ac.token, ARRAY[]::text[])) AS doc_len
            FROM article_chunks ac
            WHERE ac.doc_id IS NOT NULL
            ON CONFLICT (chunk_id) DO UPDATE
            SET doc_id = EXCLUDED.doc_id,
                doc_len = EXCLUDED.doc_len;
            """
        )

        # 2) Collection stats (n_docs, avg_doc_len)
        cur.execute(
            """
            INSERT INTO chunk_collection_stats (id, n_docs, avg_doc_len)
            SELECT
                1,
                COUNT(*) AS n_docs,
                COALESCE(AVG(doc_len)::float, 0) AS avg_doc_len
            FROM chunk_stats
            ON CONFLICT (id) DO UPDATE
            SET n_docs      = EXCLUDED.n_docs,
                avg_doc_len = EXCLUDED.avg_doc_len;
            """
        )

        # 3) Per (chunk, token) tf
        cur.execute(
            """
            INSERT INTO chunk_term_freq (chunk_id, token, tf)
            SELECT
                ac.id AS chunk_id,
                tok   AS token,
                COUNT(*) AS tf
            FROM article_chunks ac
            CROSS JOIN LATERAL unnest(COALESCE(ac.token, ARRAY[]::text[])) AS tok
            WHERE ac.doc_id IS NOT NULL
            GROUP BY ac.id, tok
            ON CONFLICT (chunk_id, token) DO UPDATE
            SET tf = EXCLUDED.tf;
            """
        )

        # 4) Global term stats (df + idf)
        cur.execute(
            """
            INSERT INTO chunk_term_stats (token, df, idf)
            SELECT
                t.token,
                t.df,
                ln((c.n_docs - t.df + 0.5) / (t.df + 0.5) + 1) AS idf
            FROM (
                SELECT token, COUNT(*) AS df
                FROM (
                    SELECT DISTINCT chunk_id, token
                    FROM chunk_term_freq
                ) d
                GROUP BY token
            ) t
            CROSS JOIN chunk_collection_stats c
            WHERE c.id = 1
            ON CONFLICT (token) DO UPDATE
            SET df  = EXCLUDED.df,
                idf = EXCLUDED.idf;
            """
        )

        # Summaries
        cur.execute("SELECT COUNT(*) FROM chunk_stats;")
        chunk_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunk_term_freq;")
        tf_rows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunk_term_stats;")
        vocab_size = cur.fetchone()[0]
        cur.execute("SELECT n_docs, avg_doc_len FROM chunk_collection_stats WHERE id = 1;")
        coll = cur.fetchone()

        conn.commit()
        print(
            "[chunk-bm25] Done. chunk_stats rows="
            f"{chunk_count}, chunk_term_freq rows={tf_rows}, vocab={vocab_size}, "
            f"n_docs={coll[0] if coll else 'n/a'}, avg_doc_len={coll[1] if coll else 'n/a'}"
        )

    except Exception as e:
        conn.rollback()
        print("[chunk-bm25] ERROR building chunk BM25 tables:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("[chunk-bm25] DB connection closed.")


def main():
    parser = argparse.ArgumentParser(description="Precompute BM25 stats for chunk-level retrieval.")
    parser.add_argument(
        "--no-truncate",
        action="store_true",
        help="Skip truncating tables; upsert instead (may leave stale rows if chunks were deleted).",
    )
    args = parser.parse_args()
    rebuild_chunk_bm25(truncate=not args.no_truncate)


if __name__ == "__main__":
    main()
