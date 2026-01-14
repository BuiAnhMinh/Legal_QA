from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from json.decoder import JSONDecodeError
from typing import List, Tuple, Optional
import re
import sys
from pathlib import Path

import numpy as np
from psycopg2.extras import execute_values
from tqdm import tqdm

# Ensure project root is on sys.path when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.config import (
    BATCH_SIZE,
    MAX_CHARS,
    EMBED_CONCURRENCY,
    SAVE_EVERY,
    get_client,
    get_connection,
)

DEFAULT_MODEL_NAME = "baai/bge-m3"


def combine_chunk_title(title: str | None, text: str) -> str:
    """
    Prefix chunk text with its title when available to give the encoder more context.
    """
    prefix = (title or "").strip()
    if not prefix:
        return text
    return f"{prefix}\n\n{text}"


def _clean(val: str | None) -> str:
    return (val or "").strip()


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


def _embed_one_batch(model: str, rows: List[Tuple[int, str]]) -> Tuple[List[Tuple[int, str]], dict]:
    """
    rows: [(row_id, text), ...] where row_id is chunk id (or article id if target=articles)
    returns: ([(row_id, pgvector_literal), ...], usage_dict)
    """
    ids = [r[0] for r in rows]
    texts = preprocess_batch([r[1] for r in rows])

    resp = _embed_call_with_retry(model=model, texts=texts)
    embs = [np.array(d.embedding, dtype="float32") for d in resp.data]
    literals = [_emb_to_pgvector_literal(e) for e in embs]
    usage = {}
    usage_obj = getattr(resp, "usage", None)
    if usage_obj:
        # OpenAI/OpenRouter style usage fields
        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        total_tokens = getattr(usage_obj, "total_tokens", None)
        if hasattr(usage_obj, "to_dict"):
            data = usage_obj.to_dict()
            prompt_tokens = prompt_tokens or data.get("prompt_tokens")
            total_tokens = total_tokens or data.get("total_tokens")
        if isinstance(usage_obj, dict):
            prompt_tokens = prompt_tokens or usage_obj.get("prompt_tokens")
            total_tokens = total_tokens or usage_obj.get("total_tokens")
        usage = {
            "prompt_tokens": int(prompt_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }
    return list(zip(ids, literals)), usage


def _bulk_update_embeddings(cur, table: str, column: str, pairs: List[Tuple[int, str]]):
    """
    Bulk update:
      UPDATE <table> SET <column> = v.emb::vector FROM (VALUES ...) v(id, emb) WHERE <table>.id = v.id
    """
    if table not in {"article_chunks", "articles"}:
        raise ValueError(f"Unsupported table: {table}")
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", column):
        raise ValueError(f"Invalid column name: {column}")

    sql = f"""
        UPDATE {table} AS t
        SET {column} = v.emb::vector
        FROM (VALUES %s) AS v(id, emb)
        WHERE t.id = v.id
    """
    execute_values(cur, sql, pairs, template="(%s, %s)", page_size=2000)


def main(
    model_name: str | None = None,
    limit: int | None = None,
    target: str = "chunks",
    use_chunk_title: bool = True,
    target_column: str = "embedding_bge_m3",
    law_title_fallback: bool = True,
    law_title_only: bool = False,
    price_per_1k: Optional[float] = None,
) -> None:
    model = model_name or DEFAULT_MODEL_NAME
    table = "article_chunks" if target == "chunks" else "articles"
    include_titles = use_chunk_title and target == "chunks"

    print(f"[embed] Target table='{table}', model='{model}', batch_size={BATCH_SIZE}, concurrency={EMBED_CONCURRENCY}")

    conn = get_connection()
    cur = conn.cursor()

    try:
        params: List = []

        if table == "article_chunks":
            sql = f"""
                SELECT
                    ac.id,
                    ac.chunk_title,
                    l.title AS law_title,
                    ac.text
                FROM article_chunks ac
                JOIN articles a ON a.id = ac.article_fk
                LEFT JOIN laws l ON l.id = a.law_fk
                WHERE {target_column} IS NULL
                ORDER BY ac.id
            """
        elif target == "articles":
            sql = f"""
                SELECT
                    a.id,
                    a.title AS article_title,
                    l.title AS law_title,
                    a.article_id,
                    a.text
                FROM articles a
                LEFT JOIN laws l ON l.id = a.law_fk
                WHERE {target_column} IS NULL
                ORDER BY a.id
            """
        else:
            sql = f"""
                SELECT id, text
                FROM {table}
                WHERE {target_column} IS NULL
                ORDER BY id
            """
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        cur.execute(sql, params)
        raw_rows = cur.fetchall()

        if table == "article_chunks":
            rows: List[Tuple[int, str]] = []
            for rid, chunk_title, law_title, text in raw_rows:
                if include_titles:
                    ct = _clean(chunk_title)
                    lt = _clean(law_title) if law_title_fallback else ""
                    effective_title = lt if law_title_only else (ct or lt)
                    rows.append((rid, combine_chunk_title(effective_title, text)))
                else:
                    rows.append((rid, text))
        elif target == "articles" and law_title_fallback:
            rows = []
            for rid, article_title, law_title, article_id, text in raw_rows:
                at = _clean(article_title)
                lt = _clean(law_title)
                if law_title_only:
                    effective_title = lt
                elif at:
                    effective_title = at
                elif lt:
                    suffix = f" — Article {article_id}" if article_id else ""
                    effective_title = f"{lt}{suffix}"
                else:
                    effective_title = f"Article {article_id}" if article_id else "Untitled article"
                rows.append((rid, combine_chunk_title(effective_title, text)))
        else:
            rows = [(rid, text) for rid, text in raw_rows]

        total = len(rows)
        print(f"[embed] Rows without {target_column}: {total}")
        if total == 0:
            return

        # Build batches
        batches = [rows[i : i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

        completed = 0
        failed_batches = 0
        commit_counter = 0
        total_prompt_tokens = 0
        total_total_tokens = 0

        with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
            futures = {ex.submit(_embed_one_batch, model, b): b for b in batches}

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding (concurrent)"):
                batch_rows = futures[fut]
                try:
                    pairs, usage = fut.result()
                    _bulk_update_embeddings(cur, table=table, column=target_column, pairs=pairs)
                    completed += len(pairs)
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_total_tokens += usage.get("total_tokens", 0)
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
        print(f"[embed] Stored embeddings in {table}.{target_column}")
        if total_total_tokens:
            print(f"[embed] Usage approx: prompt_tokens={total_prompt_tokens}, total_tokens={total_total_tokens}")
            if price_per_1k is not None:
                est_cost = (total_total_tokens / 1000.0) * price_per_1k
                print(f"[embed] Estimated cost @ {price_per_1k}/1K tokens: ~{est_cost:.4f}")

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
    parser.add_argument(
        "--no-chunk-title",
        action="store_true",
        help="For chunk target, skip prefixing chunk_title before embedding (defaults to including it).",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="embedding_bge_m3",
        help="Target vector column to update (default: embedding_bge_m3).",
    )
    parser.add_argument(
        "--no-law-title-fallback",
        action="store_true",
        help="Disable using law.title as a fallback when article/chunk titles are missing.",
    )
    parser.add_argument(
        "--law-title-only",
        action="store_true",
        help="Prefix only law.title, ignore article/chunk titles.",
    )
    parser.add_argument(
        "--price-per-1k",
        type=float,
        default=None,
        help="Optional price per 1K tokens to estimate embedding cost from API usage.",
    )
    args = parser.parse_args()
    main(
        model_name=args.model,
        limit=args.limit,
        target=args.target,
        use_chunk_title=not args.no_chunk_title,
        target_column=args.column,
        law_title_fallback=not args.no_law_title_fallback,
        law_title_only=args.law_title_only,
        price_per_1k=args.price_per_1k,
    )
