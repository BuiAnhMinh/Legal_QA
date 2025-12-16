from __future__ import annotations

import argparse
import json
import time
import random
from json.decoder import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from app.config import BATCH_SIZE, MAX_CHARS, EMBED_CONCURRENCY, get_client, DATA_DIR
from app.data_loader import load_train_data

DEFAULT_MODEL = "baai/bge-m3"
DEFAULT_OUT_NPY = DATA_DIR / "train_embedding_bge_m3.npy"
DEFAULT_OUT_META = DATA_DIR / "train_embedding_meta.json"


def preprocess_batch(texts: List[str]) -> List[str]:
    processed = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        t = t.lower()
        if len(t) > MAX_CHARS:
            t = t[:MAX_CHARS]
        processed.append(t)
    return processed


def _embed_call(model: str, texts: List[str]):
    client = get_client()
    return client.embeddings.create(model=model, input=texts)


def _embed_call_with_retry(model: str, texts: List[str], max_attempts: int = 5):
    delay = 1.0
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return _embed_call(model, texts)
        except JSONDecodeError as e:
            last_err = e
        except Exception as e:  # broad catch for transient HTTP/OpenRouter errors
            last_err = e

        if attempt < max_attempts:
            sleep_s = delay + random.random()
            time.sleep(sleep_s)
            delay = min(delay * 2, 20.0)

    raise last_err  # type: ignore[misc]


def _embed_one_batch(model: str, rows: List[Tuple[int, str]]) -> Tuple[List[int], np.ndarray]:
    ids = [r[0] for r in rows]
    texts = preprocess_batch([r[1] for r in rows])
    resp = _embed_call_with_retry(model=model, texts=texts)
    embs = np.array([d.embedding for d in resp.data], dtype="float32")
    return ids, embs


def embed_questions(model: str, questions: List[dict]) -> Tuple[np.ndarray, List[str]]:
    rows = [(idx, q["text"]) for idx, q in enumerate(questions)]
    batches = [rows[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

    results: List[Tuple[int, np.ndarray]] = []
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
        futures = {ex.submit(_embed_one_batch, model, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding train"):
            ids, embs = fut.result()
            results.append((ids[0], embs))

    # Results may be out of order; reassemble by original idx
    ordered = sorted(results, key=lambda x: x[0])
    emb_list = [emb for _, emb in ordered]
    final = np.concatenate(emb_list, axis=0)
    if final.shape[0] != len(questions):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(questions)}, got {final.shape[0]}"
        )
    qids = [q["question_id"] for q in questions]
    return final, qids


def main(model: str, limit: int | None, out_npy: Path, out_meta: Path) -> None:
    questions = load_train_data()
    if limit:
        questions = questions[:limit]
    print(f"[cache-train] Questions to embed: {len(questions)}")

    start = time.monotonic()
    embeddings, qids = embed_questions(model, questions)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, embeddings)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta = {"question_ids": qids, "model": model, "shape": list(embeddings.shape)}
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.monotonic() - start
    print(f"[cache-train] Saved embeddings -> {out_npy} | meta -> {out_meta} in {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache train.json embeddings to .npy for BGE-M3 cosine search.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Embedding model (default: baai/bge-m3).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_NPY,
        help="Output .npy path (default: data/train_embedding_bge_m3.npy).",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=DEFAULT_OUT_META,
        help="Output metadata JSON with question_id order.",
    )
    args = parser.parse_args()
    main(model=args.model, limit=args.limit, out_npy=args.out, out_meta=args.meta)
