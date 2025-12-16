from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from underthesea import word_tokenize

from app.data_loader import load_train_data

# Default cache paths (produced by scripts/cache_train_embeddings.py)
DEFAULT_EMB_PATH = Path("data/train_embedding_bge_m3.npy")
DEFAULT_META_PATH = Path("data/train_embedding_meta.json")


def tokenize_question(text: str) -> List[str]:
    tok_str = word_tokenize(text or "", format="text")
    return [t.lower() for t in tok_str.split() if t]


def fbeta_score(gold: Set[int], pred: Sequence[int], beta: float = 2.0) -> float:
    gold = set(gold or [])
    pred_set = set(pred)
    if not gold and not pred_set:
        return 1.0

    tp = len(gold & pred_set)
    fp = len(pred_set - gold)
    fn = len(gold - pred_set)
    if tp == 0:
        return 0.0

    beta2 = beta * beta
    return (1 + beta2) * tp / ((1 + beta2) * tp + beta2 * fn + fp)


def precision_recall(gold: Set[int], pred: Sequence[int]) -> tuple[float, float]:
    gold = set(gold or [])
    pred_set = set(pred)
    tp = len(gold & pred_set)
    fp = len(pred_set - gold)
    fn = len(gold - pred_set)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return prec, rec


def load_cached_embeddings(
    emb_path: Path = DEFAULT_EMB_PATH, meta_path: Path = DEFAULT_META_PATH
) -> Tuple[np.ndarray, List[str]]:
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {emb_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Embedding metadata not found: {meta_path}")

    embs = np.load(emb_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    qids = meta.get("question_ids") or []
    if len(qids) != embs.shape[0]:
        raise ValueError(
            f"Embedding rows ({embs.shape[0]}) != question_ids in meta ({len(qids)})"
        )
    return embs, qids


def load_questions_with_embeddings(
    limit: int | None = None,
    emb_path: Path = DEFAULT_EMB_PATH,
    meta_path: Path = DEFAULT_META_PATH,
) -> List[Dict]:
    questions = load_train_data()
    embeddings, qids = load_cached_embeddings(emb_path=emb_path, meta_path=meta_path)

    # Build lookup question_id -> question dict
    q_lookup = {q["question_id"]: q for q in questions}
    ordered_questions: List[Dict] = []
    for idx, qid in enumerate(qids):
        q = q_lookup.get(qid)
        if not q:
            continue
        q_copy = dict(q)
        q_copy["embedding"] = embeddings[idx]
        ordered_questions.append(q_copy)

    if limit is not None:
        ordered_questions = ordered_questions[:limit]

    if not ordered_questions:
        raise ValueError("No questions matched embeddings. Check meta question_ids.")

    return ordered_questions
