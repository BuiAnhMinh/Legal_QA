from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from json.decoder import JSONDecodeError
from typing import List, Tuple

import numpy as np
from psycopg2.extras import execute_values
from tqdm import tqdm

from app.config import (
    BATCH_SIZE,
    MAX_CHARS,
    EMBED_CONCURRENCY,
    SAVE_EVERY,
    get_client,
    get_connection,
)

DEFAULT_MODEL_NAME = "baai/bge-m3"


def preprocess_batch(texts: List[str]) -> List[str]:
    """
    Lowercase and truncate to MAX_CHARS for safety.
    (Chunks should already be <= CHUNK_MAX_CHARS, but keep this guard.)
    """
    processed = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        t = t.lower()
        if len(t) > MAX_CHARS:
            t = t[:MAX_CHARS]
        processed.append(t)
    return processed


def _emb_to_pgvector_literal(emb: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in emb.tolist()) + "]"


def _embed_call_with_retry(model: str, texts: List[str], max_attempts: int = 6):
    """
    Retry on JSONDecodeError / network hiccups / transient 5xx/429-like issues.
    We intentionally create a fresh client per call (thread-safe).
    """
    delay = 1.0
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = get_client()
            return client.embeddings.create(model=model, input=texts)
        except JSONDecodeError as e:
            last_err = e
        except Exception as e:
            # OpenRouter / httpx errors often land here; retry with backoff.
            last_err = e

        if attempt < max_attempts:
            sleep_s = delay + random.random()
            time.sleep(sleep_s)
            delay = min(delay * 2, 20.0)

    raise last_err  # type: ignore[misc]


def _embed_one_batch(model: str, rows: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    rows: [(row_id, text), ...] where row_id is chunk id (or article id if target=articles)
    returns: [(row_id, pgvector_literal), ...]
    """
    ids = [r[0] for r in rows]
    texts = preprocess_batch([r[1] for r in rows])

    resp = _embed_call_with_retry(model=model, texts=texts)
    embs = [np.array(d.embedding, dtype="float32") for d in resp.data]
    literals = [_emb_to_pgvector_literal(e) for e in embs]
    return list(zip(ids, literals))


def _bulk_update_embeddings(cur, table: str, pairs: List[Tuple[int, str]]):
    """
    Bulk update:
      UPDATE <table> SET embedding_bge_m3 = v.emb::vector FROM (VALUES ...) v(id, emb) WHERE <table>.id = v.id
    """
    sql = f"""
        UPDATE {table} AS t
        SET embedding_bge_m3 = v.emb::vector
        FROM (VALUES %s) AS v(id, emb)
        WHERE t.id = v.id
    """
    execute_values(cur, sql, pairs, template="(%s, %s)", page_size=2000)


def main(model_name: str | None = None, limit: int | None = None, target: str = "chunks") -> None:
    model = model_name or DEFAULT_MODEL_NAME
    table = "article_chunks" if target == "chunks" else "articles"

    print(f"[embed] Target table='{table}', model='{model}', batch_size={BATCH_SIZE}, concurrency={EMBED_CONCURRENCY}")

    conn = get_connection()
    cur = conn.cursor()

    try:
        sql = f"""
            SELECT id, text
            FROM {table}
            WHERE embedding_bge_m3 IS NULL
            ORDER BY id
        """
        params: List = []
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        cur.execute(sql, params)
        rows: List[Tuple[int, str]] = cur.fetchall()
        total = len(rows)
        print(f"[embed] Rows without embedding_bge_m3: {total}")
        if total == 0:
            return

        # Build batches
        batches = [rows[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

        completed = 0
        failed_batches = 0
        commit_counter = 0

        with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
            futures = {ex.submit(_embed_one_batch, model, b): b for b in batches}

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding (concurrent)"):
                batch_rows = futures[fut]
                try:
                    pairs = fut.result()
                    _bulk_update_embeddings(cur, table=table, pairs=pairs)
                    completed += len(pairs)
                    commit_counter += 1

                    if commit_counter >= max(1, SAVE_EVERY):
                        conn.commit()
                        commit_counter = 0

                except Exception as e:
                    failed_batches += 1
                    # Don’t crash the entire run; log and continue
                    bad_ids = [r[0] for r in batch_rows]
                    print(f"\n[embed] ERROR batch (ids {bad_ids[:3]}...): {repr(e)}")

        conn.commit()
        print(f"[embed] DONE. updated rows={completed}, failed_batches={failed_batches}")
        print(f"[embed] Stored embeddings in {table}.embedding_bge_m3")

    except Exception as e:
        conn.rollback()
        print("[embed] FATAL ERROR:", repr(e))
        raise
    finally:
        cur.close()
        conn.close()
        print("[embed] DB connection closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill missing embeddings using BGE-M3 via OpenRouter (concurrent).")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name (default: baai/bge-m3).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to embed.")
    parser.add_argument("--target", type=str, default="chunks", choices=["chunks", "articles"], help="Embed table: chunks (recommended) or articles (legacy).")
    args = parser.parse_args()
    main(model_name=args.model, limit=args.limit, target=args.target)
