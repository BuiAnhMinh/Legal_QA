"""Inspect BM25, dense, and hybrid ranks for a single question."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple

import asyncpg

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.retrieval_shared import (
    ann_query,
    bm25_query,
    hybrid_score_map,
    init_pgvector,
    tokenize_and_bm25_terms,
)
from app.semantic_eval_utils import load_questions_with_embeddings


def _rank_lookup(pairs: List[Tuple[int, float]]) -> Dict[int, int]:
    """Return doc_id -> rank (1-based)."""
    return {doc: idx + 1 for idx, (doc, _) in enumerate(pairs)}


async def main_async(args: argparse.Namespace) -> None:
    questions = load_questions_with_embeddings(
        limit=None, emb_path=args.emb_path, meta_path=args.meta_path
    )
    q_map = {q.get("question_id"): q for q in questions}
    question = q_map.get(args.question_id)
    if not question:
        raise SystemExit(f"question_id '{args.question_id}' not found in embeddings/meta.")

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=max(1, args.concurrency),
        init=init_pgvector,
    )

    try:
        bm25_pairs = await bm25_query(
            pool=pool,
            query_terms=tokenize_and_bm25_terms(question["text"]),
            top_k=args.bm25_top,
        )
        dense_pairs = await ann_query(
            pool=pool,
            query_vec=question["embedding"],
            top_k=args.top_k,
            chunk_limit=args.dense_chunks,
            probes=args.probes,
            ef_search=args.ef_search,
            metric=args.metric,
            column=args.column,
        )
    finally:
        await pool.close()

    # For cosine/L2, ann_query returns distances; flip sign to treat as similarity like main script.
    dense_pairs_sim = [
        (doc, -score) if args.metric in ("cosine", "l2") else (doc, score)
        for doc, score in dense_pairs
    ]

    bm25_rank = _rank_lookup(bm25_pairs)
    dense_rank = _rank_lookup(dense_pairs_sim)

    hybrid_scores = hybrid_score_map(bm25_pairs, dense_pairs_sim, alpha=args.alpha)
    hybrid_sorted = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    hybrid_rank = {doc: idx + 1 for idx, (doc, _) in enumerate(hybrid_sorted)}

    gold_ids = sorted(set(int(x) for x in question.get("gold_doc_ids", [])))

    print(f"question_id={args.question_id}")
    print(f"text={question.get('text')}\n")
    print(
        f"bm25_top={args.bm25_top}, dense_chunks={args.dense_chunks}, top_k={args.top_k}, "
        f"alpha={args.alpha}, metric={args.metric}, column={args.column}"
    )
    print("doc_id\tbm25_score\tbm25_rank\tdense_score\tdense_rank\thybrid_score\thybrid_rank")
    for doc_id in gold_ids:
        b_score = next((s for d, s in bm25_pairs if d == doc_id), None)
        d_score = next((s for d, s in dense_pairs_sim if d == doc_id), None)
        h_score = hybrid_scores.get(doc_id)
        print(
            f"{doc_id}\t"
            f"{b_score if b_score is not None else 'NA'}\t"
            f"{bm25_rank.get(doc_id, 'NA')}\t"
            f"{d_score if d_score is not None else 'NA'}\t"
            f"{dense_rank.get(doc_id, 'NA')}\t"
            f"{h_score if h_score is not None else 'NA'}\t"
            f"{hybrid_rank.get(doc_id, 'NA')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect BM25/dense/hybrid ranks for one question_id."
    )
    parser.add_argument("--question-id", required=True, help="Question ID to inspect.")
    parser.add_argument("--top-k", type=int, default=100, help="Final top-K for dense query.")
    parser.add_argument("--bm25-top", type=int, default=200, help="BM25 doc candidates.")
    parser.add_argument(
        "--dense-chunks",
        type=int,
        default=500,
        help="ANN chunk candidates before doc aggregation.",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="BM25 weight.")
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "ip"],
        help="Distance/score metric for ANN.",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="embedding_with_title_bge_m3",
        help="Chunk embedding column to query.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="DB pool size (also limits concurrent queries; unused beyond pool).",
    )
    parser.add_argument("--probes", type=int, default=None, help="ivfflat.probes (optional).")
    parser.add_argument("--ef-search", type=int, default=None, help="hnsw.ef_search (optional).")
    parser.add_argument(
        "--emb-path",
        type=Path,
        default=Path("data/train_embedding_bge_m3.npy"),
        help="Question embedding .npy path.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("data/train_embedding_meta.json"),
        help="Question meta JSON path.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
