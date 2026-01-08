"""Single-question HyQE evaluation (question expansions -> averaged embedding -> ANN)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Iterable, List, Sequence

import asyncpg

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, LLM_MODEL, get_client
from app.data_loader import load_train_data
from app.retrieval_shared import ann_query, init_pgvector
from app.semantic_eval_utils import fbeta_score, precision_recall
from database.db_embedding import DEFAULT_MODEL_NAME as DEFAULT_CHUNK_MODEL


def _generate_hyqe(question: str, count: int, max_chars: int) -> List[str]:
    """
    Generate multiple legal-style query expansions for the question.
    Each line is a rewritten query focusing on legal terminology and citations.
    """
    prompt = (
        "Viết các câu hỏi/đề mục truy vấn pháp lý (tiếng Việt) diễn đạt lại câu hỏi sau. "
        "Dùng ngôn ngữ pháp lý, điều khoản, tên văn bản, và biến thể từ khóa. "
        f"Tạo {count} dòng, mỗi dòng một truy vấn ngắn.\n\n"
        f"Câu hỏi: {question}"
    )
    client = get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Bạn viết các truy vấn pháp lý súc tích bằng tiếng Việt."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    text = (resp.choices[0].message.content or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        if ln and len(out) < count:
            out.append(ln[:max_chars])
    # Fallback to include the original question if generation fails.
    if not out:
        out = [question[:max_chars]]
    return out


def _embed_texts(model: str, texts: List[str]) -> List[List[float]]:
    client = get_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _mean_vec(vecs: Iterable[Sequence[float]]) -> List[float]:
    vecs = list(vecs)
    if not vecs:
        return []
    dim = len(vecs[0])
    sums = [0.0] * dim
    for v in vecs:
        for i, val in enumerate(v):
            sums[i] += float(val)
    return [s / len(vecs) for s in sums]


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
    expansions: int,
) -> None:
    hyqe_queries = _generate_hyqe(question_text, count=expansions, max_chars=max_chars)
    hyqe_embs = _embed_texts(emb_model, hyqe_queries)
    query_vec = _mean_vec(hyqe_embs)

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
            query_vec=query_vec,
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
    print("HyQE queries:")
    for q in hyqe_queries:
        print(" -", q)
    print(f"Gold doc_ids: {sorted(gold_set)}")
    if rank is None:
        print(f"Gold not found in top {top_k}.")
    else:
        print(f"Gold rank: {rank} / {top_k}")
    print(f"precision={p:.4f} | recall={r:.4f} | f2={f2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-question HyQE eval (query expansions + ANN).")
    parser.add_argument("--qid", type=str, default=None, help="Question id (e.g. vlsp_123).")
    parser.add_argument("--question", type=str, default=None, help="Raw question text.")
    parser.add_argument("--doc-id", type=int, default=None, help="Gold doc_id override.")
    parser.add_argument(
        "--emb-model",
        type=str,
        default=DEFAULT_CHUNK_MODEL,
        help="Embedding model to encode HyQE queries (should match DB column dims, default: baai/bge-m3).",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--chunk-limit", type=int, default=2000)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2", "ip"])
    parser.add_argument("--probes", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--expansions", type=int, default=5, help="Number of HyQE rewrites to average.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Unused (accepted for CLI compatibility with hybrid scripts).",
    )
    parser.add_argument(
        "--emb-path",
        type=Path,
        default=Path("data/train_embedding_bge_m3.npy"),
        help="Question embedding .npy path (only used when loading by qid).",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("data/train_embedding_meta.json"),
        help="Question meta JSON path (only used when loading by qid).",
    )
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
            expansions=args.expansions,
        )
    )


if __name__ == "__main__":
    main()
