"""Hybrid BM25 + ANN retrieval using HyQE query expansions for dense queries."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import asyncpg
import numpy as np

from app.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER, LLM_MODEL, get_client
from app.data_loader import load_train_data
from app.retrieval_shared import ann_query, bm25_query, hybrid_merge, init_pgvector
from app.semantic_eval_utils import fbeta_score, precision_recall, tokenize_question
from database.db_embedding import DEFAULT_MODEL_NAME as DEFAULT_CHUNK_MODEL

SOURCE_FILTER = "tvpl"


def _generate_hyqe(question: str, count: int, max_chars: int) -> List[str]:
    """
    Generate multiple legal-style query expansions for the question.
    Each line is a rewritten query focusing on legal terminology and citations.
    """
    prompt = (
        "Viết lại câu hỏi sau thành các truy vấn pháp lý ngắn (6–14 từ), tiếng Việt, dạng từ khóa. "
        "Mỗi dòng một truy vấn, không markdown, không đánh số, không giải thích dài dòng. "
        "Ưu tiên thuật ngữ pháp lý, tên văn bản, điều/ khoản/ điểm liên quan.\n\n"
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


def _weighted_mean_vec(vecs: List[Sequence[float]], weights: List[float]) -> List[float]:
    if not vecs or not weights or len(vecs) != len(weights):
        return []
    dim = len(vecs[0])
    sums = [0.0] * dim
    w_sum = 0.0
    for v, w in zip(vecs, weights):
        w_sum += w
        for i, val in enumerate(v):
            sums[i] += float(val) * w
    if w_sum == 0:
        return []
    return [s / w_sum for s in sums]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _to_similarity(score: float, metric: str) -> float:
    """
    Convert ANN distance/score to a similarity-like value for merging.
    Cosine/L2 distances are negated; inner product is used as-is.
    """
    if metric in ("cosine", "l2"):
        return -score
    return score


async def _build_hyqe_vectors(
    question_text: str,
    emb_model: str,
    expansions: int,
    max_chars: int,
    semaphore: asyncio.Semaphore | None,
) -> tuple[List[tuple[str, List[float], float]], List[str]]:
    """
    Generate HyQE rewrites and build per-rewrite embeddings + weights.
    We return ([(query, vector, weight), ...], query_texts_for_logging).
    Runs in a thread to avoid blocking the event loop and guards concurrency with a semaphore.
    """

    def _run():
        try:
            queries = _generate_hyqe(question_text, count=expansions, max_chars=max_chars)
        except Exception as e:
            print(f"[hyqe] generation failed; using raw question. err={e}")
            queries = [question_text[:max_chars]]

        # Drop LLM preamble lines (e.g., "Dưới đây là ..."), strip numbering, dedupe, and ensure raw question is included.
        cleaned: List[str] = []
        seen = set()
        for q in queries:
            q_strip = q.strip()
            if not q_strip:
                continue
            if q_strip.lower().startswith("dưới đây"):
                continue
            if q_strip[:2].isdigit():  # remove leading numbering like "1. ..." or "2) ..."
                q_strip = q_strip.lstrip("0123456789. )")
                q_strip = q_strip.strip()
            if q_strip in seen:
                continue
            seen.add(q_strip)
            cleaned.append(q_strip[:max_chars])

        raw_q = question_text.strip()[:max_chars]
        if raw_q and raw_q not in seen:
            cleaned.append(raw_q)

        if not cleaned:
            cleaned = [question_text[:max_chars]]
        try:
            # Embed raw question + expansions together for similarity gating.
            texts = [raw_q] + cleaned
            embs = _embed_texts(emb_model, texts)
        except Exception as e:
            print(f"[hyqe] embedding failed; using raw-only. err={e}")
            return [(raw_q, [], 1.0)], [raw_q]

        raw_emb = embs[0]
        exp_embs = embs[1:]
        pairs = list(zip(cleaned, exp_embs))

        # Keep only expansions close to the original to reduce drift.
        gate = 0.65  # slightly looser gate to retain more rewrites
        filtered = []
        for q, v in pairs:
            cos = _cosine(raw_emb, v)
            if cos >= gate:
                filtered.append((q, v, cos))
        if not filtered:
            filtered = [(q, v, _cosine(raw_emb, v)) for q, v in pairs]  # fallback: keep all if gate drops everything

        # Take top-m most similar expansions.
        filtered.sort(key=lambda x: x[2], reverse=True)
        top_m = 6
        filtered = filtered[:top_m]

        # Weight rewrites by their cosine to the raw question; anchor raw at 0.6.
        raw_weight = 0.6
        rem = max(0.0, 1.0 - raw_weight)
        cos_sum = sum(max(c, 0.0) for _, _, c in filtered)
        per_exp_weights: List[float] = []
        if filtered:
            if cos_sum > 0:
                per_exp_weights = [rem * (max(c, 0.0) / cos_sum) for _, _, c in filtered]
            else:
                per_exp_weights = [rem / len(filtered)] * len(filtered)

        hyqe_candidates: List[tuple[str, List[float], float]] = [(raw_q, raw_emb, raw_weight)]
        for (q, v, _), w in zip(filtered, per_exp_weights):
            hyqe_candidates.append((q, v, w))

        query_texts = [q for q, _, _ in hyqe_candidates]
        return hyqe_candidates, query_texts

    if semaphore is None:
        return await asyncio.to_thread(_run)
    async with semaphore:
        return await asyncio.to_thread(_run)


async def evaluate(
    pool: asyncpg.pool.Pool,
    questions: Iterable[Dict],
    top_k: int,
    bm25_top: int,
    dense_chunks: int,
    alpha: float,
    concurrency: int,
    probes: int | None,
    ef_search: int | None,
    metric: str,
    emb_model: str,
    expansions: int,
    max_chars: int,
    hyqe_concurrency: int,
    dump_path: Path | None,
    column: str,
    progress_every: int,
    dump_flush: int,
) -> float:
    q_list = list(questions)
    if not q_list:
        print("No questions to evaluate.")
        return 0.0

    q_iter = iter(q_list)
    lock = asyncio.Lock()
    dump_rows: List[Dict] = []
    dump_lock = asyncio.Lock()
    hyqe_sem = asyncio.Semaphore(max(1, hyqe_concurrency)) if hyqe_concurrency else None
    processed = 0
    total = len(q_list)
    dump_written = 0

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("", encoding="utf-8")

    async def flush_dump_buffer(force: bool = False) -> None:
        nonlocal dump_written
        if dump_path is None:
            return
        if not dump_rows:
            return
        if not force and len(dump_rows) < dump_flush:
            return
        async with dump_lock:
            if not dump_rows:
                return
            with dump_path.open("a", encoding="utf-8") as f:
                for item in dump_rows:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            dump_written += len(dump_rows)
            dump_rows.clear()

    async def next_item():
        async with lock:
            return next(q_iter, None)

    async def runner():
        nonlocal processed
        f2s: List[float] = []
        ps: List[float] = []
        rs: List[float] = []

        while True:
            q = await next_item()
            if q is None:
                break

            hyqe_candidates, hyqe_queries = await _build_hyqe_vectors(
                question_text=q["text"],
                emb_model=emb_model,
                expansions=expansions,
                max_chars=max_chars,
                semaphore=hyqe_sem,
            )

            bm25_pairs = await bm25_query(
                pool=pool,
                query_terms=tokenize_question(q["text"]),
                top_k=bm25_top,
                source=SOURCE_FILTER,
            )
            dense_score_map: Dict[int, float] = {}
            for _, vec, weight in hyqe_candidates:
                if not vec:
                    continue
                dense_pairs = await ann_query(
                    pool=pool,
                    query_vec=vec,
                    top_k=top_k,
                    chunk_limit=dense_chunks,
                    probes=probes,
                    ef_search=ef_search,
                    metric=metric,
                    column=column,
                    source=SOURCE_FILTER,
                )
                for doc, score in dense_pairs:
                    sim = _to_similarity(score, metric)
                    dense_score_map[doc] = dense_score_map.get(doc, 0.0) + weight * sim
            dense_pairs_sim = list(dense_score_map.items())
            dense_pairs_sim.sort(key=lambda x: x[1], reverse=True)

            hybrid_docs = hybrid_merge(bm25_pairs, dense_pairs_sim, alpha=alpha)[:top_k]

            f2_val = fbeta_score(q["gold_doc_ids"], hybrid_docs, beta=2.0)
            p, r = precision_recall(q["gold_doc_ids"], hybrid_docs)
            f2s.append(f2_val)
            ps.append(p)
            rs.append(r)

            async with lock:
                processed += 1
                count = processed
            if progress_every > 0 and count % progress_every == 0:
                print(f"[progress] processed {count}/{total} questions")

            if dump_path is not None:
                dump_rows.append(
                    {
                        "question_id": q.get("question_id"),
                        "text": q.get("text"),
                        "hyqe_queries": hyqe_queries,
                        "gold_all": sorted(list(q["gold_doc_ids"])),
                        "pred_docs": hybrid_docs,
                        "precision": p,
                        "recall": r,
                        "f2": f2_val,
                    }
                )
                await flush_dump_buffer()

        return f2s, ps, rs

    worker_n = max(1, concurrency)
    workers = [asyncio.create_task(runner()) for _ in range(worker_n)]
    parts = await asyncio.gather(*workers)

    await flush_dump_buffer(force=True)

    f2_scores = [x for part in parts for x in part[0]]
    prec_scores = [x for part in parts for x in part[1]]
    rec_scores = [x for part in parts for x in part[2]]

    macro_f2 = float(np.mean(f2_scores)) if f2_scores else 0.0
    macro_prec = float(np.mean(prec_scores)) if prec_scores else 0.0
    macro_rec = float(np.mean(rec_scores)) if rec_scores else 0.0

    if dump_path is not None:
        print(f"Wrote {dump_written} question rows to {dump_path}")

    print(
        f"Hybrid BM25+HyQE-ANN @ {top_k}: alpha={alpha:.2f}, bm25_top={bm25_top}, "
        f"dense_chunks={dense_chunks}, expansions={expansions}, macro F2={macro_f2:.4f} | "
        f"macro Precision={macro_prec:.4f} | macro Recall={macro_rec:.4f} "
        f"over {len(f2_scores)} questions"
    )
    return macro_f2


async def main_async(args: argparse.Namespace) -> None:
    questions = load_train_data()
    if args.limit is not None:
        questions = questions[: args.limit]

    print(
        f"Evaluating Hybrid BM25 + HyQE ANN on {len(questions)} questions "
        f"(top_k={args.top_k}, bm25_top={args.bm25_top}, dense_chunks={args.dense_chunks}, "
        f"alpha={args.alpha}, expansions={args.expansions}, emb_model={args.emb_model}, "
        f"concurrency={args.concurrency}, hyqe_concurrency={args.hyqe_concurrency}, "
        f"probes={args.probes}, ef_search={args.ef_search}, metric={args.metric}, column={args.column})."
    )

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
        await evaluate(
            pool=pool,
            questions=questions,
            top_k=args.top_k,
            bm25_top=args.bm25_top,
            dense_chunks=args.dense_chunks,
            alpha=args.alpha,
            concurrency=args.concurrency,
            probes=args.probes,
            ef_search=args.ef_search,
            metric=args.metric,
            emb_model=args.emb_model,
            expansions=args.expansions,
            max_chars=args.max_chars,
            hyqe_concurrency=args.hyqe_concurrency,
            dump_path=args.dump_misses,
            column=args.column,
            progress_every=args.progress_every,
            dump_flush=args.dump_flush,
        )
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + ANN (chunk-level) retrieval using HyQE expansions for dense queries.",
    )
    parser.add_argument("--top-k", type=int, default=500, help="Final top-K documents to evaluate.")
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
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 in hybrid score (dense weight = 1 - alpha).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent DB queries; also limits DB pool size.",
    )
    parser.add_argument(
        "--hyqe-concurrency",
        type=int,
        default=1,
        help="Limit concurrent HyQE generation/embedding calls to avoid rate limits.",
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
        "--emb-model",
        type=str,
        default=DEFAULT_CHUNK_MODEL,
        help="Embedding model to encode HyQE queries (must match DB column dims).",
    )
    parser.add_argument(
        "--expansions",
        type=int,
        default=5,
        help="Number of HyQE rewrites to average.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Max characters per HyQE rewrite and embedding text.",
    )
    parser.add_argument(
        "--dump-misses",
        type=Path,
        default=None,
        help="Write per-question results (gold/preds/precision/recall/f2) to this JSONL file.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N questions (0 disables).",
    )
    parser.add_argument(
        "--dump-flush",
        type=int,
        default=50,
        help="Flush dump file every N rows (only when --dump-misses is set).",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
