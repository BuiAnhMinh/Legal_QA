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

ANCHORS = [
    "người đại diện",
    "phạm vi đại diện",
    "giao dịch dân sự",
    "ủy quyền",
]

FORBIDDEN = [
    "kiểm tra viên",
    "thanh tra",
    "điện lực",
    "công an",
    "kiểm lâm",
    "quản lý thị trường",
    "biên bản kiểm tra",
]

STYLE_HINTS = [
    "Giải thích đơn giản cho người không chuyên, câu ngắn.",
    "Diễn đạt theo văn phong pháp lý trung tính, súc tích.",
    "Nhấn mạnh giới hạn và phạm vi đại diện cùng căn cứ xác định.",
    "Nhấn mạnh xung đột lợi ích: không giao dịch với chính mình hoặc đại diện đôi.",
    "Nhấn mạnh nghĩa vụ thông báo phạm vi đại diện cho bên giao dịch.",
]

MIN_W, MAX_W = 100, 150

PROMPT_TEMPLATE = (
    "You are a Vietnamese legal expert and lawyer.\n\n"
    "Your task is to generate a hypothetical explanatory document that describes\n"
    "the legal rule, legal principle, or legal mechanism expressed in the given law article.\n\n"
    "ANCHOR REQUIREMENT:\n"
    '- The output MUST contain all of these phrases (verbatim): "người đại diện", "phạm vi đại diện", "giao dịch dân sự", "ủy quyền".\n'
    "- Do NOT introduce any new titles/roles not present in the input.\n\n"
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
    "- Length: 100–150 words. If your output is not between 100 and 150 words, it is invalid and must be regenerated.\n"
    "- Structure: clear sentences, no bullet points.\n"
    "- If you produce any bullet points or numbering, your output is invalid.\n"
    "- Do not include headings or lists.\n"
    "- Do not use Markdown (no bold, no lists, no headings).\n\n"
    "STYLE: {style_hint}\n\n"
    "INPUT LAW ARTICLE:\n"
    "{law_article}"
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _valid_len(text: str) -> bool:
    w = _word_count(text)
    return MIN_W <= w <= MAX_W


def _no_markdown(text: str) -> bool:
    return ("**" not in text) and ("#" not in text) and ("```" not in text)


def _generate_hypo_doc(
    article_text: str, max_chars: int, temperature: float, style_hint: str
) -> str:
    trimmed = article_text[:max_chars]
    prompt = PROMPT_TEMPLATE.format(law_article=trimmed, style_hint=style_hint)
    client = get_client()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You produce concise Vietnamese legal explanations."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        presence_penalty=0.6,
        frequency_penalty=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def _is_good_hypo(text: str) -> bool:
    t = text.lower()
    if any(a not in t for a in ANCHORS):
        return False
    if any(f in t for f in FORBIDDEN):
        return False
    if not _valid_len(text):
        return False
    if not _no_markdown(text):
        return False
    if re.search(r"^\s*[-•\d]+[.)]?", text, flags=re.MULTILINE):
        return False
    return True


def _generate_hypo_with_retry(
    article_text: str, max_chars: int, temperature: float, style_hint: str, tries: int = 6
) -> str:
    last = ""
    for _ in range(max(1, tries)):
        out = _generate_hypo_doc(
            article_text, max_chars=max_chars, temperature=temperature, style_hint=style_hint
        )
        if _is_good_hypo(out):
            return out
        last = out
    return last


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


def _sim(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return _cosine(a, b)
    if metric == "ip":
        return _ip(a, b)
    # l2: convert to similarity by negating distance
    return -_l2(a, b)


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

    preview = article_text[:300].replace("\n", " ")
    print(f"ARTICLE PREVIEW: {preview}")

    style_seq = STYLE_HINTS[:num_hypo] if num_hypo <= len(STYLE_HINTS) else STYLE_HINTS
    if num_hypo > len(STYLE_HINTS):
        extra_needed = num_hypo - len(STYLE_HINTS)
        style_seq = STYLE_HINTS + STYLE_HINTS[:extra_needed]

    async def gen_one(style_hint: str) -> str:
        return await asyncio.to_thread(
            _generate_hypo_with_retry,
            article_text,
            max_article_chars,
            temperature,
            style_hint,
            6,
        )

    hypo_docs = await asyncio.gather(*(gen_one(s) for s in style_seq))

    embeddings = _embed_texts(emb_model, [question, article_text] + hypo_docs)
    q_emb = np.array(embeddings[0], dtype="float32")
    a_emb = np.array(embeddings[1], dtype="float32")
    hypo_embs = [np.array(e, dtype="float32") for e in embeddings[2:]]

    scored = []
    for idx, (text, emb) in enumerate(zip(hypo_docs, hypo_embs), start=1):
        s_q = _sim(q_emb, emb, metric)
        s_a = _sim(a_emb, emb, metric)
        score = 0.7 * s_q + 0.3 * s_a
        scored.append((idx, score, s_q, s_a, text))

    reverse = True  # similarity; higher better (we negate l2 above)
    scored.sort(key=lambda x: x[1], reverse=reverse)

    print(f"Doc_id: {doc_id}")
    print(f"Question: {question}")
    print(f"Article chars: {len(article_text)} (used {min(len(article_text), max_article_chars)})")
    print(f"Metric: {metric} (higher is better)" if metric != "l2" else "Metric: l2 (lower is better)")

    for idx, score, s_q, s_a, text in scored:
        wc = _word_count(text)
        print("-" * 80)
        print(f"Hypo #{idx} | words={wc} | score={score:.4f} | s_q={s_q:.4f} | s_a={s_a:.4f}")
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
