from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Set, Tuple, Optional

import asyncpg
from underthesea import word_tokenize

from app.config import (
    STOPWORDS_PATH,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_MIN_CHARS,
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME,
)

CHUNK_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS article_chunks (
    id                  BIGSERIAL PRIMARY KEY,
    article_fk          INT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    doc_id              INT,
    chunk_title         TEXT,
    chunk_index         INT NOT NULL,          -- 1..N
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

CREATE INDEX IF NOT EXISTS idx_article_chunks_missing_emb
    ON article_chunks(id)
    WHERE embedding_bge_m3 IS NULL;
"""

_WORKER_STOPWORDS: Optional[Set[str]] = None


def _init_worker(stopwords_list: List[str]) -> None:
    global _WORKER_STOPWORDS
    _WORKER_STOPWORDS = set(stopwords_list)


def load_stopwords(path: Path) -> Set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Stopwords file not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def tokenize(text: str) -> tuple[List[str], List[str]]:
    """
    Uses global worker stopwords for speed (no re-sending set each task).
    """
    global _WORKER_STOPWORDS
    stopwords = _WORKER_STOPWORDS or set()

    tok_str = word_tokenize(text, format="text")
    tokens = [t.lower() for t in tok_str.split() if t]
    tokens_no_stop = [t for t in tokens if t not in stopwords]
    return tokens, tokens_no_stop


def _find_split_point(text: str, start: int, end: int, lookback: int = 200) -> int:
    if end >= len(text):
        return end
    lo = max(start + 1, end - lookback)
    window = text[lo:end]
    idx = window.rfind(" ")
    return end if idx == -1 else lo + idx


def chunk_text(text: str, max_chars: int, overlap: int, min_chars: int) -> List[Tuple[int, int, str]]:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return []

    n = len(text)
    if n <= max_chars:
        return [(0, n, text)]

    chunks: List[Tuple[int, int, str]] = []
    start = 0

    while start < n:
        end = min(n, start + max_chars)
        end2 = _find_split_point(text, start, end)

        if end2 - start < min_chars:
            end2 = end

        piece = text[start:end2].strip()
        if piece:
            chunks.append((start, end2, piece))

        if end2 >= n:
            break

        start = max(0, end2 - overlap)

        # safety
        if chunks and start <= chunks[-1][0]:
            start = chunks[-1][1]

    if not chunks:
        chunks = [(0, min(n, max_chars), text[:max_chars])]

    return chunks


def process_article(
    article_pk: int,
    doc_id: Optional[int],
    text: str,
    article_title: Optional[str],
) -> List[Tuple]:
    """
    Runs in ProcessPool (CPU parallel).
    Returns rows for COPY/INSERT:
      (article_fk, doc_id, chunk_title, chunk_index, char_start, char_end, text, token, token_no_stopword)
    """
    pieces = chunk_text(
        text=text or "",
        max_chars=CHUNK_MAX_CHARS,
        overlap=CHUNK_OVERLAP_CHARS,
        min_chars=CHUNK_MIN_CHARS,
    )
    if not pieces:
        return []

    total = len(pieces)

    def _build_chunk_title(idx: int) -> Optional[str]:
        base = (article_title or "").strip()
        if not base:
            return None
        if total > 1:
            return f"{base} (chunk {idx}/{total})"
        return base

    out: List[Tuple] = []
    for idx, (s, e, ctext) in enumerate(pieces, start=1):
        toks, toks_ns = tokenize(ctext)
        out.append(
            (
                article_pk,
                doc_id,
                _build_chunk_title(idx),
                idx,
                s,
                e,
                ctext,
                toks,
                toks_ns,
            )
        )
    return out


async def ensure_schema(pool: asyncpg.Pool, rebuild: bool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(CHUNK_TABLE_SQL)
        await conn.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS chunk_title TEXT;")
        await conn.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_with_title_bge_m3 vector(1024);")
        await conn.execute("ALTER TABLE article_chunks ADD COLUMN IF NOT EXISTS embedding_bge_m3 vector(1024);")
        await conn.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS title TEXT;")
        if rebuild:
            print("[chunk-async] TRUNCATE article_chunks ...")
            await conn.execute("TRUNCATE TABLE article_chunks RESTART IDENTITY;")


async def insert_rows_copy(pool: asyncpg.Pool, rows: List[Tuple]) -> None:
    """
    Fast path (requires rebuild/truncate mode): COPY into table.
    """
    async with pool.acquire() as conn:
        await conn.copy_records_to_table(
            "article_chunks",
            records=rows,
            columns=[
                "article_fk",
                "doc_id",
                "chunk_title",
                "chunk_index",
                "char_start",
                "char_end",
                "text",
                "token",
                "token_no_stopword",
            ],
        )


async def insert_rows_upsert(pool: asyncpg.Pool, rows: List[Tuple]) -> None:
    """
    Slower path (append mode): INSERT .. ON CONFLICT.
    """
    sql = """
    INSERT INTO article_chunks
      (article_fk, doc_id, chunk_title, chunk_index, char_start, char_end, text, token, token_no_stopword)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
    ON CONFLICT (article_fk, chunk_index) DO UPDATE
    SET doc_id = EXCLUDED.doc_id,
        chunk_title = EXCLUDED.chunk_title,
        char_start = EXCLUDED.char_start,
        char_end = EXCLUDED.char_end,
        text = EXCLUDED.text,
        token = EXCLUDED.token,
        token_no_stopword = EXCLUDED.token_no_stopword;
    """
    async with pool.acquire() as conn:
        await conn.executemany(sql, rows)


async def rebuild_chunks_async(
    limit: int | None,
    rebuild: bool,
    workers: int,
    prefetch: int,
    insert_batch_size: int,
    max_inflight: int,
) -> None:
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"[chunk-async] Loaded stopwords={len(stopwords)} from {STOPWORDS_PATH}")

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=4,
    )

    try:
        await ensure_schema(pool, rebuild=rebuild)

        insert_fn = insert_rows_copy if rebuild else insert_rows_upsert

        # ProcessPool for CPU-heavy Underthesea tokenization
        proc_pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(list(stopwords),),
        )

        inserted_chunks = 0
        processed_articles = 0
        buffer: List[Tuple] = []

        sem = asyncio.Semaphore(max_inflight)
        loop = asyncio.get_running_loop()

        async def handle_one(record) -> None:
            nonlocal inserted_chunks, processed_articles, buffer
            article_pk = record["id"]
            doc_id = record["doc_id"]
            text = record["text"]
            title = record["title"]

            async with sem:
                rows = await loop.run_in_executor(
                    proc_pool, process_article, article_pk, doc_id, text, title
                )

            processed_articles += 1
            if rows:
                buffer.extend(rows)

            # flush if buffer large
            if len(buffer) >= insert_batch_size:
                await insert_fn(pool, buffer)
                inserted_chunks += len(buffer)
                buffer.clear()

                if processed_articles % 2000 == 0:
                    print(f"[chunk-async] processed_articles={processed_articles}, inserted_chunks={inserted_chunks}")

        async with pool.acquire() as conn:
            sql = "SELECT id, doc_id, text, title FROM articles ORDER BY id"
            args = []
            if limit is not None:
                sql += " LIMIT $1"
                args = [limit]

            # async cursor streaming
            stmt = await conn.prepare(sql)
            cursor = stmt.cursor(*args, prefetch=prefetch)

            tasks: set[asyncio.Task] = set()
            async for rec in cursor:
                t = asyncio.create_task(handle_one(rec))
                tasks.add(t)

                # keep task set bounded
                if len(tasks) >= max_inflight * 2:
                    done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    # propagate errors early
                    for d in done:
                        d.result()

            # finish remaining tasks
            if tasks:
                done, _ = await asyncio.wait(tasks)
                for d in done:
                    d.result()

        # final flush
        if buffer:
            await insert_fn(pool, buffer)
            inserted_chunks += len(buffer)
            buffer.clear()

        print(f"[chunk-async] DONE. processed_articles={processed_articles}, inserted_chunks={inserted_chunks}")

        async with pool.acquire() as conn:
            total_chunks = await conn.fetchval("SELECT COUNT(*) FROM article_chunks;")
            print(f"[chunk-async] article_chunks rows = {total_chunks}")

        proc_pool.shutdown(wait=True)

    finally:
        await pool.close()
        print("[chunk-async] DB pool closed.")


def main():
    parser = argparse.ArgumentParser(description="Async + parallel chunking/tokenization into article_chunks.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-rebuild", action="store_true", help="Append/upsert instead of truncating and COPY.")
    parser.add_argument("--workers", type=int, default=6, help="CPU workers for Underthesea + chunking.")
    parser.add_argument("--prefetch", type=int, default=200, help="DB cursor prefetch rows.")
    parser.add_argument("--insert-batch", type=int, default=5000, help="Rows to buffer before inserting.")
    parser.add_argument("--max-inflight", type=int, default=24, help="Max concurrent CPU tasks in flight.")
    args = parser.parse_args()

    asyncio.run(
        rebuild_chunks_async(
            limit=args.limit,
            rebuild=not args.no_rebuild,
            workers=args.workers,
            prefetch=args.prefetch,
            insert_batch_size=args.insert_batch,
            max_inflight=args.max_inflight,
        )
    )


if __name__ == "__main__":
    main()
