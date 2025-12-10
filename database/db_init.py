from typing import Dict, List, Set

from underthesea import word_tokenize
from pathlib import Path
from app.config import get_connection
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
    text        TEXT NOT NULL,
    embedding   vector(1536),
    tokens      TEXT[],
    source      TEXT NOT NULL DEFAULT 'unknown', 
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    is_amending_article BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT uq_law_article UNIQUE (law_id, article_id)
);

CREATE INDEX IF NOT EXISTS idx_articles_law_fk
    ON articles(law_fk);

CREATE INDEX IF NOT EXISTS idx_articles_law_id_article_id
    ON articles(law_id, article_id);
"""

def main():
    docs = load_law_documents()
    total_docs = len(docs)
    print(f"Loaded {total_docs} docs from loader.")

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(SCHEMA_SQL)
        conn.commit()
        print("Schema ensured (laws, articles).")

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

        # Insert articles with tokens
        inserted_articles = 0
        print(f"Start inserting {total_docs} articles...")
        for d in docs:
            law_id = d["law_id"]
            if law_id not in law_id_to_pk:
                print(f"Skipping article with unknown law_id={law_id}")
                continue

            law_pk = law_id_to_pk[law_id]
            text = d["text"] or ""
            tokens = _tokenize_text(text)

            source = d.get("source", "unknown")
            
            cur.execute(
                """
                INSERT INTO articles (law_fk, law_id, article_id, text, tokens, source)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (law_id, article_id) DO UPDATE
                SET law_fk = EXCLUDED.law_fk,
                    text = EXCLUDED.text,
                    tokens = EXCLUDED.tokens,
                    source = EXCLUDED.source;
                """,
                (law_pk, law_id, d["article_id"], text, tokens, source),
            )
            inserted_articles += 1

            if inserted_articles % 5000 == 0:
                print(f"Inserted {inserted_articles}/{total_docs} articles so far...")
                conn.commit()

        conn.commit()
        print(f"Finished inserting ~{inserted_articles} articles (out of {total_docs}).")

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
