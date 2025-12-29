from typing import Dict, List, Set

from underthesea import word_tokenize
from pathlib import Path
from app.config import STOPWORDS_PATH, get_connection
from app.data_loader import load_law_documents

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS laws (
    id          SERIAL PRIMARY KEY,
    law_id      TEXT NOT NULL UNIQUE,
    title       TEXT,
    source      TEXT NOT NULL DEFAULT 'unknown',
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS articles (
    id          SERIAL PRIMARY KEY,
    law_fk      INTEGER NOT NULL REFERENCES laws(id) ON DELETE CASCADE,
    law_id      TEXT NOT NULL,
    article_id  TEXT NOT NULL,
    article_idx TEXT,
    title       TEXT,
    text        TEXT NOT NULL,
    embedding   vector(1536),
    embedding_bge_m3 vector(1024),
    token       TEXT[],
    token_no_stopword TEXT[],
    source      TEXT NOT NULL DEFAULT 'unknown',
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    is_amending_article BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT uq_law_article UNIQUE (law_id, article_id)
);

-- Chunks: every article has 1..N chunks
CREATE TABLE IF NOT EXISTS article_chunks (
    id                  BIGSERIAL PRIMARY KEY,
    article_fk          INT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    doc_id              INT,
    chunk_title         TEXT,
    chunk_index         INT NOT NULL,
    char_start          INT NOT NULL,
    char_end            INT NOT NULL,
    text                TEXT NOT NULL,
    token               TEXT[],
    token_no_stopword   TEXT[],
    embedding_bge_m3    vector(1024),
    embedding_with_title_bge_m3 vector(1024),
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_article_chunk UNIQUE (article_fk, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_article_chunks_article_fk
    ON article_chunks(article_fk);

CREATE INDEX IF NOT EXISTS idx_article_chunks_doc_id
    ON article_chunks(doc_id);

-- Drop BM25 helper tables if they exist so we always start clean
DROP TABLE IF EXISTS article_stats CASCADE;
DROP TABLE IF EXISTS article_term_freq CASCADE;
DROP TABLE IF EXISTS term_stats CASCADE;
DROP TABLE IF EXISTS collection_stats CASCADE;

-- 1) Per-document stats (length etc.) – keyed by doc_id
CREATE TABLE article_stats (
    article_id   INT PRIMARY KEY,
    doc_len      INT NOT NULL
);

-- 2) Per (doc, term) stats: tf – keyed by doc_id + token
CREATE TABLE article_term_freq (
    article_id   INT NOT NULL,
    token        TEXT NOT NULL,
    tf           INT NOT NULL,
    PRIMARY KEY (article_id, token)
);

-- 3) Global term stats: df + idf
CREATE TABLE term_stats (
    token        TEXT PRIMARY KEY,
    df           INT NOT NULL,
    idf          DOUBLE PRECISION NOT NULL
);

-- 4) Collection-level stats: n_docs, avgdl
CREATE TABLE collection_stats (
    id           INT PRIMARY KEY CHECK (id = 1),
    n_docs       INT NOT NULL,
    avg_doc_len  DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_law_fk
    ON articles(law_fk);

CREATE INDEX IF NOT EXISTS idx_articles_law_id_article_id
    ON articles(law_id, article_id);

-- Helpful index to quickly find postings by token
CREATE INDEX IF NOT EXISTS idx_article_term_token
    ON article_term_freq (token, article_id);

-- Global doc id from original corpus (VLSP / legal_corpus.json)
ALTER TABLE articles
ADD COLUMN IF NOT EXISTS doc_id INT UNIQUE;
"""


def load_stopwords(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        stops = {line.strip().lower() for line in f if line.strip()}
    print(f"Loaded {len(stops)} stopwords from {path}")
    return stops


def _tokenize_text(text: str, stopwords: Set[str]) -> tuple[List[str], List[str]]:
    if not isinstance(text, str):
        text = str(text)
    tok_str = word_tokenize(text, format="text")
    # Lowercase to make indexing/querying case-insensitive; keep stopwords in `tokens`.
    tokens = [t.lower() for t in tok_str.split() if t]
    tokens_no_stop = [t for t in tokens if t not in stopwords]
    return tokens, tokens_no_stop


def main():
    stopwords = load_stopwords(STOPWORDS_PATH)

    docs = load_law_documents()
    total_docs = len(docs)
    print(f"Loaded {total_docs} docs from loader.")

    conn = get_connection()
    cur = conn.cursor()

    try:
        # Create schema and helper columns
        cur.execute(SCHEMA_SQL)
        # Ensure token columns exist even if table was created in an older version
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS token TEXT[];")
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS token_no_stopword TEXT[];")
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS article_idx TEXT;")
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS title TEXT;")
        # Ensure embedding columns exist (legacy tables may be missing bge_m3)
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding vector(1536);")
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector(1024);")
        cur.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_with_title_bge_m3 vector(1024);")
        cur.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector(1024);")
        cur.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS chunk_title TEXT;")
        conn.commit()
        print("Schema ensured (laws, articles, BM25 tables).")

        # Insert distinct laws
        seen = set()
        for d in docs:
            law_id = d["law_id"]
            if law_id in seen:
                continue
            seen.add(law_id)

            source = d.get("source", "unknown")

            cur.execute(
                """
                INSERT INTO laws (law_id, source)
                VALUES (%s, %s)
                ON CONFLICT (law_id) DO UPDATE
                SET source = EXCLUDED.source;
                """,
                (law_id, source),
            )
        conn.commit()
        print(f"Inserted/ensured {len(seen)} laws.")

        # Build mapping law_id -> laws.id
        cur.execute("SELECT id, law_id FROM laws;")
        rows = cur.fetchall()
        law_id_to_pk: Dict[str, int] = {law_id: pk for pk, law_id in rows}
        print(f"Loaded {len(law_id_to_pk)} law_id -> id mappings.")

        missing_laws = {d["law_id"] for d in docs if d["law_id"] not in law_id_to_pk}
        if missing_laws:
            print("WARNING: some law_id not in laws table, examples:", list(missing_laws)[:5])

        # Insert articles with tokens + doc_id
        inserted_articles = 0
        print(f"Start inserting {total_docs} articles...")
        for d in docs:
            law_id = d["law_id"]
            if law_id not in law_id_to_pk:
                print(f"Skipping article with unknown law_id={law_id}")
                continue

            law_pk = law_id_to_pk[law_id]
            text = d.get("text") or ""
            tokens, tokens_no_stop = _tokenize_text(text, stopwords)

            source = d.get("source", "unknown")

            # Try to get global doc_id from corpus; fallback to d["id"] if present
            raw_doc_id = d.get("doc_id")
            if raw_doc_id is None and "id" in d:
                raw_doc_id = d["id"]
            doc_id = raw_doc_id  # can be None; those rows will be excluded from BM25 stats

            cur.execute(
                """
                INSERT INTO articles (
                    law_fk, law_id, article_id, doc_id, text, token, token_no_stopword, source
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (law_id, article_id) DO UPDATE
                SET law_fk            = EXCLUDED.law_fk,
                    text              = EXCLUDED.text,
                    token             = EXCLUDED.token,
                    token_no_stopword = EXCLUDED.token_no_stopword,
                    source            = EXCLUDED.source,
                    doc_id            = EXCLUDED.doc_id;
                """,
                (law_pk, law_id, d["article_id"], doc_id, text, tokens, tokens_no_stop, source),
            )
            inserted_articles += 1

            if inserted_articles % 5000 == 0:
                print(f"Inserted {inserted_articles}/{total_docs} articles so far...")
                conn.commit()

        conn.commit()
        print(f"Finished inserting ~{inserted_articles} articles (out of {total_docs}).")

        # Quick sanity check on doc_id coverage
        cur.execute("SELECT COUNT(*) FROM articles WHERE doc_id IS NOT NULL;")
        non_null_docid = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM articles WHERE doc_id IS NULL;")
        null_docid = cur.fetchone()[0]
        print(f"Articles with non-null doc_id: {non_null_docid}")
        print(f"Articles with NULL doc_id (excluded from BM25 index): {null_docid}")

        print("Recomputing BM25 stats (article_stats, article_term_freq, term_stats, collection_stats)...")

        # 1) Per-document length (only docs with a valid global doc_id)
        cur.execute(
            """
            INSERT INTO article_stats (article_id, doc_len)
            SELECT
                a.doc_id,
                cardinality(COALESCE(a.token, ARRAY[]::text[])) AS doc_len
            FROM articles a
            WHERE a.doc_id IS NOT NULL;
            """
        )

        # 2) Per (doc, token) term frequency (same doc_id filter)
        cur.execute(
            """
            INSERT INTO article_term_freq (article_id, token, tf)
            SELECT
                a.doc_id,
                tok,
                COUNT(*) AS tf
            FROM articles a
            CROSS JOIN LATERAL unnest(COALESCE(a.token, ARRAY[]::text[])) AS tok
            WHERE a.doc_id IS NOT NULL
            GROUP BY a.doc_id, tok;
            """
        )

        # 3) Collection stats (robust even if article_stats is empty)
        cur.execute(
            """
            INSERT INTO collection_stats (id, n_docs, avg_doc_len)
            SELECT
                1,
                COUNT(*) AS n_docs,
                COALESCE(AVG(doc_len)::float, 0) AS avg_doc_len
            FROM article_stats
            ON CONFLICT (id) DO UPDATE
            SET n_docs      = EXCLUDED.n_docs,
                avg_doc_len = EXCLUDED.avg_doc_len;
            """
        )

        # 4) Term stats (df + idf)
        cur.execute(
            """
            INSERT INTO term_stats (token, df, idf)
            SELECT
                t.token,
                t.df,
                ln((c.n_docs - t.df + 0.5) / (t.df + 0.5) + 1) AS idf
            FROM (
                SELECT token, COUNT(*) AS df
                FROM (
                    SELECT DISTINCT article_id, token
                    FROM article_term_freq
                ) d
                GROUP BY token
            ) t
            CROSS JOIN collection_stats c
            WHERE c.id = 1
            ON CONFLICT (token) DO UPDATE
            SET df  = EXCLUDED.df,
                idf = EXCLUDED.idf;
            """
        )

        conn.commit()
        print("BM25 stats recomputed.")

    except Exception as e:
        conn.rollback()
        print("ERROR during migration:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("DB connection closed.")


if __name__ == "__main__":
    main()
