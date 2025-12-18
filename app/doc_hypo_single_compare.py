from __future__ import annotations

import argparse
import asyncio
import re
from typing import List

import asyncpg
import numpy as np

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

PROMPT_TEMPLATE = (
    "You are a Vietnamese legal expert and lawyer.\n\n"
    "Your task is to generate a hypothetical explanatory document that describes\n"
    "the legal rule, legal principle, or legal mechanism expressed in the given law article.\n\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Do NOT restate, paraphrase, or answer any specific user question.\n"
    "- Do NOT mention specific legal actions, disputes, cases, or factual scenarios\n"
    "  (e.g., inheritance, asset division, contracts, lawsuits).\n"
    "- Explain the rule at an abstract and general level only.\n"
    "- Focus on legal concepts, scope of authority, rights, obligations, and limitations.\n"
    "- Use neutral, formal but plain Vietnamese legal language suitable for a layperson.\n"
    "- Do NOT introduce new legal conditions or interpretations not present in the article.\n"
    "- Avoid rare synonyms; prefer commonly used Vietnamese legal terminology.\n"
    "- Do NOT mention article numbers or citations.\n\n"
    "OUTPUT REQUIREMENTS:\n"
    "- Write one coherent explanatory document.\n"
    "- Length: 100–150 words.\n"
    "- Structure: clear sentences, no bullet points.\n"
    "- Do not include headings or lists.\n\n"
    "INPUT LAW ARTICLE:\n"
    "{law_article}"
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _generate_hypo_doc(article_text: str, max_chars: int, temperature: float) -> str:
    trimmed = article_text[:max_chars]
    prompt = PROMPT_TEMPLATE.format(law_article=trimmed)
    client = get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You produce concise Vietnamese legal explanations."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _embed_texts(model: str, texts: List[str]) -> List[List[float]]:
    client = get_client()
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _ip(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


async def _fetch_article_text(pool: asyncpg.pool.Pool, doc_id: int) -> str:
    sql = "SELECT text FROM articles WHERE doc_id = $1;"
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, doc_id)
    if not row or not row["text"]:
        raise ValueError(f"Article not found for doc_id={doc_id}")
    return str(row["text"])


async def main_async(
    doc_id: int,
    question: str,
    num_hypo: int,
    emb_model: str,
    metric: str,
    max_article_chars: int,
    temperature: float,
) -> None:
    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=1,
    )

    try:
        article_text = await _fetch_article_text(pool, doc_id)
    finally:
        await pool.close()

    hypo_docs = [
        _generate_hypo_doc(article_text, max_chars=max_article_chars, temperature=temperature)
        for _ in range(num_hypo)
    ]

    embeddings = _embed_texts(emb_model, [question] + hypo_docs)
    q_emb = np.array(embeddings[0], dtype="float32")
    hypo_embs = [np.array(e, dtype="float32") for e in embeddings[1:]]

    scored = []
    for idx, (text, emb) in enumerate(zip(hypo_docs, hypo_embs), start=1):
        if metric == "cosine":
            score = _cosine(q_emb, emb)
        elif metric == "ip":
            score = _ip(q_emb, emb)
        else:
            score = _l2(q_emb, emb)
        scored.append((idx, score, text))

    reverse = metric != "l2"
    scored.sort(key=lambda x: x[1], reverse=reverse)

    print(f"Doc_id: {doc_id}")
    print(f"Question: {question}")
    print(f"Article chars: {len(article_text)} (used {min(len(article_text), max_article_chars)})")
    print(f"Metric: {metric} (higher is better)" if metric != "l2" else "Metric: l2 (lower is better)")

    for idx, score, text in scored:
        wc = _word_count(text)
        print("-" * 80)
        print(f"Hypo #{idx} | words={wc} | score={score:.4f}")
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate hypothetical docs from one article and compare to a query."
    )
    parser.add_argument("--doc-id", type=int, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--num-hypo", type=int, default=3)
    parser.add_argument("--emb-model", type=str, default=EMB_MODEL)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "ip", "l2"])
    parser.add_argument("--max-article-chars", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    asyncio.run(
        main_async(
            doc_id=args.doc_id,
            question=args.question,
            num_hypo=args.num_hypo,
            emb_model=args.emb_model,
            metric=args.metric,
            max_article_chars=args.max_article_chars,
            temperature=args.temperature,
        )
    )


if __name__ == "__main__":
    main()
