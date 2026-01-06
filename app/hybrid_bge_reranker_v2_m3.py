from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import asyncpg
import numpy as np
from huggingface_hub import InferenceClient

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from app.retrieval_shared import ann_query, bm25_query, hybrid_score_map, init_pgvector
from app.semantic_eval_utils import (
    fbeta_score,
    load_questions_with_embeddings,
    precision_recall,
    tokenize_question,
)


def _novita_rerank(
    client: InferenceClient,
    model: str,
    query: str,
    documents: List[Dict[str, str]],
) -> List[Dict]:
    # Novita-hosted bge-reranker is exposed via the HF router; we call via InferenceClient
    results: List[Dict] = []
    for idx, doc in enumerate(documents):
        # text_classification returns a list of label/score dicts; use the first score
        out = client.text_classification(model=model, text=query, text_pair=doc["text"])
        if not out:
            continue
        score = out[0].get("score")
        if score is None:
            continue
        results.append({"index": idx, "relevance_score": float(score)})
    return results


async def _novita_rerank_async(
    client: InferenceClient,
    model: str,
    query: str,
    documents: List[Dict[str, str]],
) -> List[Dict]:
    return await asyncio.to_thread(_novita_rerank, client, model, query, documents)


async def _fetch_doc_texts(pool: asyncpg.pool.Pool, doc_ids: Sequence[int]) -> Dict[int, str]:
    if not doc_ids:
        return {}
    sql = "SELECT doc_id, text FROM articles WHERE doc_id = ANY($1::int[]);"
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, list(doc_ids))
    return {int(r["doc_id"]): r["text"] for r in rows if r["text"]}


async def _hybrid_candidates(
    pool: asyncpg.pool.Pool,
    question_text: str,
    query_vec: Sequence[float],
    bm25_top: int,
    dense_docs: int,
    dense_chunks: int,
    alpha: float,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    hybrid_top: int,
) -> List[int]:
    tokens = tokenize_question(question_text)
    bm25_pairs = await bm25_query(pool=pool, query_terms=tokens, top_k=bm25_top)
    dense_pairs = await ann_query(
        pool=pool,
        query_vec=query_vec,
        top_k=dense_docs,
        chunk_limit=dense_chunks,
        probes=probes,
        ef_search=ef_search,
        metric=metric,
    )
    score_map = hybrid_score_map(bm25_pairs, dense_pairs, alpha=alpha)
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:hybrid_top]]


async def evaluate(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    bm25_top: int,
    dense_docs: int,
    dense_chunks: int,
    alpha: float,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    hybrid_top: int,
    rerank_top: int,
    max_chars: int,
    novita_client: InferenceClient,
    hf_model: str,
    concurrency: int,
) -> float:
    q_list = list(questions)
    if not q_list:
        print("No questions to evaluate.")
        return 0.0

    q_iter = iter(q_list)
    lock = asyncio.Lock()

    async def next_item():
        async with lock:
            return next(q_iter, None)

    async def runner():
        f2s: List[float] = []
        ps: List[float] = []
        rs: List[float] = []

        while True:
            q = await next_item()
            if q is None:
                break

            candidates = await _hybrid_candidates(
                pool=pool,
                question_text=q["text"],
                query_vec=q["embedding"],
                bm25_top=bm25_top,
                dense_docs=dense_docs,
                dense_chunks=dense_chunks,
                alpha=alpha,
                probes=probes,
                ef_search=ef_search,
                metric=metric,
                hybrid_top=hybrid_top,
            )
            doc_texts = await _fetch_doc_texts(pool, candidates)

            documents: List[Dict[str, str]] = []
            doc_index_to_id: List[int] = []
            for doc_id in candidates:
                text = doc_texts.get(doc_id)
                if not text:
                    continue
                documents.append({"text": text[:max_chars]})
                doc_index_to_id.append(doc_id)

            if not documents:
                preds: List[int] = []
            else:
                results = await _novita_rerank_async(
                    client=novita_client,
                    model=hf_model,
                    query=q["text"],
                    documents=documents,
                )
                results.sort(key=lambda r: r.get("relevance_score", 0), reverse=True)
                preds = []
                for r in results:
                    idx = r.get("index")
                    if idx is None:
                        continue
                    if 0 <= idx < len(doc_index_to_id):
                        preds.append(doc_index_to_id[idx])
                    if len(preds) >= rerank_top:
                        break

            f2s.append(fbeta_score(q["gold_doc_ids"], preds, beta=2.0))
            p, r = precision_recall(q["gold_doc_ids"], preds)
            ps.append(p)
            rs.append(r)

        return f2s, ps, rs

    worker_n = max(1, concurrency)
    workers = [asyncio.create_task(runner()) for _ in range(worker_n)]
    parts = await asyncio.gather(*workers)

    f2_scores = [x for part in parts for x in part[0]]
    prec_scores = [x for part in parts for x in part[1]]
    rec_scores = [x for part in parts for x in part[2]]

    macro_f2 = float(np.mean(f2_scores)) if f2_scores else 0.0
    macro_prec = float(np.mean(prec_scores)) if prec_scores else 0.0
    macro_rec = float(np.mean(rec_scores)) if rec_scores else 0.0

    print(
        f"Hybrid+Novita rerank @ {rerank_top}: macro F2={macro_f2:.4f} | "
        f"macro Precision={macro_prec:.4f} | macro Recall={macro_rec:.4f} "
        f"over {len(f2_scores)} questions"
    )
    return macro_f2


async def main_async(
    top_k: int,
    bm25_top: int,
    dense_docs: int,
    dense_chunks: int,
    alpha: float,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    hybrid_top: int,
    max_chars: int,
    novita_client: InferenceClient,
    hf_model: str,
    limit: int | None,
    concurrency: int,
    emb_path: Path,
    meta_path: Path,
):
    questions = load_questions_with_embeddings(
        limit=limit,
        emb_path=emb_path,
        meta_path=meta_path,
    )
    print(
        f"Hybrid+Novita rerank on {len(questions)} questions "
        f"(hybrid_top={hybrid_top}, rerank_top={top_k}, concurrency={concurrency})."
    )

    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=max(1, concurrency),
        init=init_pgvector,
    )

    try:
        await evaluate(
            pool=pool,
            questions=questions,
            bm25_top=bm25_top,
            dense_docs=dense_docs,
            dense_chunks=dense_chunks,
            alpha=alpha,
            probes=probes,
            ef_search=ef_search,
            metric=metric,
            hybrid_top=hybrid_top,
            rerank_top=top_k,
            max_chars=max_chars,
            novita_client=novita_client,
            hf_model=hf_model,
            concurrency=concurrency,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid BM25+ANN -> Novita (via Hugging Face router) rerank."
    )
    parser.add_argument("--top-k", type=int, default=100, help="Final top-K after rerank.")
    parser.add_argument("--bm25-top", type=int, default=500)
    parser.add_argument("--dense-docs", type=int, default=500)
    parser.add_argument("--dense-chunks", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2", "ip"])
    parser.add_argument("--probes", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=None)
    parser.add_argument("--hybrid-top", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--hf-model", type=str, default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--emb-path", type=Path, default=None)
    parser.add_argument("--meta-path", type=Path, default=None)
    parser.add_argument("--hf-token", type=str, default=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    args = parser.parse_args()

    if not args.hf_token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN (set env var or pass --hf-token).")

    emb_path = args.emb_path or Path("data/train_embedding_bge_m3.npy")
    meta_path = args.meta_path or Path("data/train_embedding_meta.json")

    client = InferenceClient(provider="novita", api_key=args.hf_token)

    asyncio.run(
        main_async(
            top_k=args.top_k,
            bm25_top=args.bm25_top,
            dense_docs=args.dense_docs,
            dense_chunks=args.dense_chunks,
            alpha=args.alpha,
            probes=args.probes,
            ef_search=args.ef_search,
            metric=args.metric,
            hybrid_top=args.hybrid_top,
            max_chars=args.max_chars,
            novita_client=client,
            hf_model=args.hf_model,
            limit=args.limit,
            concurrency=args.concurrency,
            emb_path=emb_path,
            meta_path=meta_path,
        )
    )


if __name__ == "__main__":
    main()
