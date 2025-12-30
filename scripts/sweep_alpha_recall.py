from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import asyncpg
import numpy as np

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.retrieval_shared import (
    ann_query,
    bm25_query,
    hybrid_merge,
    init_pgvector,
    tokenize_and_bm25_terms,
)
from app.semantic_eval_utils import (
    fbeta_score,
    load_questions_with_embeddings,
    precision_recall,
)


def split_questions(
    questions: List[Dict],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Shuffle and split questions into train/val/test (val used for tuning, test for reporting).
    """
    total = len(questions)
    rng = np.random.default_rng(seed)
    rng.shuffle(questions)

    val_n = int(total * val_fraction)
    test_n = int(total * test_fraction)

    # ensure we have at least one sample in val/test when possible
    if val_n == 0 and total > 0:
        val_n = 1
    if test_n == 0 and total - val_n > 0:
        test_n = 1
    # cap to total length
    if val_n + test_n > total:
        test_n = max(0, total - val_n)

    val = questions[:val_n]
    test = questions[val_n : val_n + test_n]
    train = questions[val_n + test_n :]
    return train, val, test


async def sweep_alphas(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    alphas: List[float],
    top_k: int,
    bm25_top: int,
    dense_chunks: int,
    metric: str,
    column: str,
    probes: int | None,
    ef_search: int | None,
    concurrency: int,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate recall/precision/F2 for multiple alphas in one pass.
    """
    q_list = list(questions)
    if not q_list:
        print("No questions to evaluate.")
        return {}

    alpha_vals = sorted(set(alphas))
    stats: Dict[float, Dict[str, List[float]]] = {
        a: {"rec": [], "prec": [], "f2": []} for a in alpha_vals
    }

    q_iter = iter(q_list)
    lock = asyncio.Lock()

    async def next_item():
        async with lock:
            return next(q_iter, None)

    async def runner():
        local: Dict[float, Dict[str, List[float]]] = {
            a: {"rec": [], "prec": [], "f2": []} for a in alpha_vals
        }
        while True:
            q = await next_item()
            if q is None:
                break

            bm25_pairs = await bm25_query(
                pool=pool,
                query_terms=tokenize_and_bm25_terms(q["text"]),
                top_k=bm25_top,
            )
            dense_pairs = await ann_query(
                pool=pool,
                query_vec=q["embedding"],
                top_k=top_k,
                chunk_limit=dense_chunks,
                probes=probes,
                ef_search=ef_search,
                metric=metric,
                column=column,
            )
            dense_pairs_sim = [
                (doc, -score) if metric in ("cosine", "l2") else (doc, score)
                for doc, score in dense_pairs
            ]

            for a in alpha_vals:
                hybrid_docs = hybrid_merge(bm25_pairs, dense_pairs_sim, alpha=a)[:top_k]
                p, r = precision_recall(q["gold_doc_ids"], hybrid_docs)
                f2 = fbeta_score(q["gold_doc_ids"], hybrid_docs, beta=2.0)
                local[a]["prec"].append(p)
                local[a]["rec"].append(r)
                local[a]["f2"].append(f2)

        return local

    worker_n = max(1, concurrency)
    workers = [asyncio.create_task(runner()) for _ in range(worker_n)]
    parts = await asyncio.gather(*workers)

    for part in parts:
        for a in alpha_vals:
            stats[a]["rec"].extend(part[a]["rec"])
            stats[a]["prec"].extend(part[a]["prec"])
            stats[a]["f2"].extend(part[a]["f2"])

    summary: Dict[float, Dict[str, float]] = {}
    for a in alpha_vals:
        recs = stats[a]["rec"]
        precs = stats[a]["prec"]
        f2s = stats[a]["f2"]
        summary[a] = {
            "recall": float(np.mean(recs)) if recs else 0.0,
            "precision": float(np.mean(precs)) if precs else 0.0,
            "f2": float(np.mean(f2s)) if f2s else 0.0,
        }

    return summary


async def evaluate_alpha(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    alpha: float,
    top_k: int,
    bm25_top: int,
    dense_chunks: int,
    metric: str,
    column: str,
    probes: int | None,
    ef_search: int | None,
    concurrency: int,
) -> Dict[str, float]:
    """
    Evaluate a single alpha on the provided questions.
    """
    results = await sweep_alphas(
        pool=pool,
        questions=questions,
        alphas=[alpha],
        top_k=top_k,
        bm25_top=bm25_top,
        dense_chunks=dense_chunks,
        metric=metric,
        column=column,
        probes=probes,
        ef_search=ef_search,
        concurrency=concurrency,
    )
    return results.get(alpha, {"recall": 0.0, "precision": 0.0, "f2": 0.0})


async def main_async(args: argparse.Namespace) -> None:
    questions = load_questions_with_embeddings(
        limit=args.limit,
        emb_path=args.emb_path,
        meta_path=args.meta_path,
    )

    train_qs, val_qs, test_qs = split_questions(
        questions=list(questions),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    print(
        f"Sweeping alphas={args.alphas} on validation set "
        f"(top_k={args.top_k}, bm25_top={args.bm25_top}, dense_chunks={args.dense_chunks}, "
        f"metric={args.metric}, column={args.column}, probes={args.probes}, ef_search={args.ef_search}). "
        f"Counts -> train:{len(train_qs)}, val:{len(val_qs)}, test:{len(test_qs)}"
    )

    if not val_qs:
        print("No validation questions available after split; aborting sweep.")
        return

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
        val_summary = await sweep_alphas(
            pool=pool,
            questions=val_qs,
            alphas=args.alphas,
            top_k=args.top_k,
            bm25_top=args.bm25_top,
            dense_chunks=args.dense_chunks,
            metric=args.metric,
            column=args.column,
            probes=args.probes,
            ef_search=args.ef_search,
            concurrency=args.concurrency,
        )
    finally:
        await pool.close()

    if not val_summary:
        return

    print("\nalpha\tRecall\tPrecision\tF2")
    for a in sorted(val_summary):
        s = val_summary[a]
        print(f"{a:.4f}\t{s['recall']:.4f}\t{s['precision']:.4f}\t{s['f2']:.4f}")

    best_alpha, best = max(
        val_summary.items(),
        key=lambda kv: (kv[1]["recall"], -kv[0]),
    )
    print(
        f"\nBest alpha by validation recall: {best_alpha:.4f} "
        f"(val recall={best['recall']:.4f}, precision={best['precision']:.4f}, f2={best['f2']:.4f})"
    )

    if test_qs:
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
            test_metrics = await evaluate_alpha(
                pool=pool,
                questions=test_qs,
                alpha=best_alpha,
                top_k=args.top_k,
                bm25_top=args.bm25_top,
                dense_chunks=args.dense_chunks,
                metric=args.metric,
                column=args.column,
                probes=args.probes,
                ef_search=args.ef_search,
                concurrency=args.concurrency,
            )
        finally:
            await pool.close()
        print(
            f"Test set @ alpha={best_alpha:.4f}: recall={test_metrics['recall']:.4f}, "
            f"precision={test_metrics['precision']:.4f}, f2={test_metrics['f2']:.4f}"
        )
    else:
        print("No test set after split; skipped test evaluation.")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep alpha to maximize recall for hybrid BM25 + ANN retrieval."
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0,1, 0.15, 0.2, 0.25, 0.3, 0.35, 0,4, 0,45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        help="Alpha values (BM25 weight) to sweep; dense weight = 1 - alpha.",
    )
    parser.add_argument("--top-k", type=int, default=500, help="Top-K documents to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions.")
    parser.add_argument(
        "--bm25-top",
        type=int,
        default=500,
        help="BM25 doc candidates.",
    )
    parser.add_argument(
        "--dense-chunks",
        type=int,
        default=2000,
        help="ANN chunk candidates before doc aggregation.",
    )
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
        default=10,
        help="Concurrent DB queries; also limits DB pool size.",
    )
    parser.add_argument(
        "--probes",
        type=int,
        default=None,
        help="Set ivfflat.probes for recall/speed tradeoff (requires IVF index).",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=None,
        help="Set hnsw.ef_search for recall/speed tradeoff (requires HNSW index).",
    )
    parser.add_argument(
        "--emb-path",
        type=Path,
        default=None,
        help="Path to question embedding .npy (default: data/train_embedding_bge_m3.npy).",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Path to question embedding meta JSON (default: data/train_embedding_meta.json).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of questions to use for validation sweep.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of questions to reserve for test reporting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling before splitting.",
    )
    args = parser.parse_args()

    emb_path = args.emb_path or Path("data/train_embedding_bge_m3.npy")
    meta_path = args.meta_path or Path("data/train_embedding_meta.json")
    args.emb_path = emb_path
    args.meta_path = meta_path

    if args.val_fraction < 0 or args.test_fraction < 0:
        raise ValueError("val_fraction and test_fraction must be non-negative.")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1.0.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
