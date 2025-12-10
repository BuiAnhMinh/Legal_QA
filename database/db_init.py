from typing import Dict, List, Set

from underthesea import word_tokenize
from pathlib import Path
from app.config import STOPWORDS_PATH, get_connection
from app.data_loader import load_law_documents

# Basic Vietnamese stopword fallback (used if STOPWORDS_PATH is missing)
DEFAULT_STOPWORDS: Set[str] = {
    "và", "là", "của", "có", "cho", "một", "các", "những", "được", "trong",
    "khi", "đã", "sẽ", "tại", "theo", "từ", "đến", "để", "bị", "bằng", "với",
    "này", "đó", "nên", "thì", "rằng", "vì", "như", "cũng", "chỉ", "rất", "ra",
    "vào", "lên", "xuống", "qua", "đi", "lại", "hơn", "trên", "dưới", "giữa",
    "khoảng", "cùng", "nhau", "người", "điều", "khoản", "không", "hoặc", "vẫn",
}

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
    token       TEXT[],
    token_no_stopword TEXT[],
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


def _load_stopwords(path: Path) -> Set[str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            stops = {line.strip().lower() for line in f if line.strip()}
        print(f"Loaded {len(stops)} stopwords from {path}")
        return stops

    print(f"Stopwords file not found at {path}, using default set ({len(DEFAULT_STOPWORDS)})")
    return DEFAULT_STOPWORDS


def _tokenize_text(text: str, stopwords: Set[str]) -> tuple[List[str], List[str]]:
    if not isinstance(text, str):
        text = str(text)
    tok_str = word_tokenize(text, format="text")
    tokens = [t for t in tok_str.split() if t]
    tokens_no_stop = [t for t in tokens if t.lower() not in stopwords]
    return tokens, tokens_no_stop


def main():
    stopwords = _load_stopwords(STOPWORDS_PATH)

    docs = load_law_documents()
    total_docs = len(docs)
    print(f"Loaded {total_docs} docs from loader.")

    conn = get_connection()
    cur = conn.cursor()

    try:
        cur.execute(SCHEMA_SQL)
        # Ensure new columns exist even if table was created previously
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS token TEXT[];")
        cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS token_no_stopword TEXT[];")
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
            tokens, tokens_no_stop = _tokenize_text(text, stopwords)

            source = d.get("source", "unknown")

            cur.execute(
                """
                INSERT INTO articles (law_fk, law_id, article_id, text, token, token_no_stopword, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (law_id, article_id) DO UPDATE
                SET law_fk = EXCLUDED.law_fk,
                    text = EXCLUDED.text,
                    token = EXCLUDED.token,
                    token_no_stopword = EXCLUDED.token_no_stopword,
                    source = EXCLUDED.source;
                """,
                (law_pk, law_id, d["article_id"], text, tokens, tokens_no_stop, source),
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
