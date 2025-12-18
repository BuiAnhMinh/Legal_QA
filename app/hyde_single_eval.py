from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List, Sequence

import asyncpg

from app.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    EMB_MODEL,
    LLM_MODEL,
    get_client,
)
from app.data_loader import load_train_data
from app.retrieval_shared import ann_query, init_pgvector
from app.semantic_eval_utils import fbeta_score, precision_recall


def _generate_hyde(question: str, max_chars: int) -> str:
    prompt = (
        "Write a concise Vietnamese legal passage that would answer the question. "
        "Use statute-style language, legal definitions, conditions, and authority terms. "
        "Do not mention that this is hypothetical. 6-10 sentences.\n\n"
        f"Question: {question}"
    )
    client = get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You write statute-style Vietnamese legal text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def _embed_texts(model: str, texts: List[str]) -> List[List[float]]:
    client = get_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _find_question(qid: str) -> dict:
    qid_norm = qid if qid.startswith("vlsp_") else f"vlsp_{qid}"
    for q in load_train_data():
        if q["question_id"] == qid_norm:
            return q
    raise ValueError(f"Question id not found: {qid_norm}")


async def _run_once(
    question_text: str,
    gold_doc_ids: Sequence[int],
    emb_model: str,
    top_k: int,
    chunk_limit: int,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    max_chars: int,
) -> None:
    hyde_text = _generate_hyde(question_text, max_chars=max_chars)
    hyde_emb = _embed_texts(emb_model, [hyde_text])[0]

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=1,
        init=init_pgvector,
    )
    try:
        pairs = await ann_query(
            pool=pool,
            query_vec=hyde_emb,
            top_k=top_k,
            chunk_limit=chunk_limit,
            probes=probes,
            ef_search=ef_search,
            metric=metric,
        )
    finally:
        await pool.close()

    preds = [doc_id for doc_id, _ in pairs]
    gold_set = set(int(x) for x in gold_doc_ids)
    p, r = precision_recall(gold_set, preds)
    f2 = fbeta_score(gold_set, preds, beta=2.0)

    rank = None
    for idx, doc_id in enumerate(preds, start=1):
        if doc_id in gold_set:
            rank = idx
            break

    print("Question:", question_text)
    print("HyDE:", hyde_text[:300].replace("\n", " ").strip())
    print(f"Gold doc_ids: {sorted(gold_set)}")
    if rank is None:
        print(f"Gold not found in top {top_k}.")
    else:
        print(f"Gold rank: {rank} / {top_k}")
    print(f"precision={p:.4f} | recall={r:.4f} | f2={f2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-question HyDE eval (ANN).")
    parser.add_argument("--qid", type=str, default=None, help="Question id (e.g. vlsp_123).")
    parser.add_argument("--question", type=str, default=None, help="Raw question text.")
    parser.add_argument("--doc-id", type=int, default=None, help="Gold doc_id override.")
    parser.add_argument("--emb-model", type=str, default=EMB_MODEL)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chunk-limit", type=int, default=2000)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2", "ip"])
    parser.add_argument("--probes", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()

    if not args.qid and not args.question:
        raise ValueError("Provide --qid or --question.")

    if args.qid:
        q = _find_question(args.qid)
        question_text = q["text"]
        gold_doc_ids = q["gold_doc_ids"]
    else:
        question_text = args.question or ""
        gold_doc_ids = []

    if args.doc_id is not None:
        gold_doc_ids = [args.doc_id]

    if not gold_doc_ids:
        raise ValueError("Missing gold doc ids. Provide --doc-id or a qid with gold.")

    asyncio.run(
        _run_once(
            question_text=question_text,
            gold_doc_ids=gold_doc_ids,
            emb_model=args.emb_model,
            top_k=args.top_k,
            chunk_limit=args.chunk_limit,
            probes=args.probes,
            ef_search=args.ef_search,
            metric=args.metric,
            max_chars=args.max_chars,
        )
    )


if __name__ == "__main__":
    main()
