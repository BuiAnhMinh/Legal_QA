from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import LAW_PATH, TRAIN_PATH


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_law_documents(
    path: Path | None = None, default_source: str = "legal_corpus"
) -> List[Dict[str, Any]]:
    """
    Flatten legal_corpus.json into a list of article docs.
    Each doc: {doc_id, law_id, article_id, text, source}
    For *this* dataset:
      - doc_id = aid (global article ID, matches train.relevant_laws)
    """
    corpus_path = Path(path) if path else LAW_PATH
    law_data = load_json(corpus_path)
    if not isinstance(law_data, list):
        raise ValueError(f"Expected list in {corpus_path}, got {type(law_data)}")

    documents: List[Dict[str, Any]] = []
    skipped_empty = 0

    for law in law_data:
        law_id = _normalize_str(law.get("law_id") or law.get("id"))
        if not law_id:
            raise ValueError(f"Missing law_id in entry: {law}")

        for artc in law.get("content", []):
            # ---- KEY CHANGE HERE ----
            aid_raw = artc.get("aid")
            if aid_raw is None:
                raise ValueError(f"Missing aid in article: {artc}")

            try:
                aid_int = int(aid_raw)
            except ValueError:
                raise ValueError(f"aid is not int-like: {aid_raw} in {artc}")

            article_id = _normalize_str(aid_int)  # just for logging / display

            text = _normalize_str(
                artc.get("content_Article") or artc.get("text") or artc.get("content")
            )
            if not text:
                skipped_empty += 1
                continue

            documents.append(
                {
                    "law_id": law_id,
                    "article_id": article_id,   # e.g. "53877"
                    "doc_id": aid_int,          # *** must match train.relevant_laws ***
                    "text": text,
                    "source": law.get("source", default_source),
                }
            )

    print(
        f"Loaded {len(documents)} documents from {corpus_path} "
        f"(skipped empty articles: {skipped_empty})."
    )
    return documents

def load_train_data(
    path: Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Normalise VLSP-style train.json:
      {
        "qid": int,
        "question": str,
        "relevant_laws": [doc_id_1, doc_id_2, ...]
      }
    into our canonical format:
      {
        "question_id": str,
        "text": str,
        "gold_doc_ids": set[int],
      }
    """
    train_path = Path(path) if path else TRAIN_PATH
    raw = load_json(train_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {train_path}, got {type(raw)}")

    out: List[Dict[str, Any]] = []
    for item in raw:
        qid = item.get("qid") or item.get("id")
        text = _normalize_str(item.get("question") or item.get("text"))
        rel = item.get("relevant_laws") or item.get("relevant_articles") or []

        out.append(
            {
                "question_id": f"vlsp_{qid}",
                "text": text,
                "gold_doc_ids": set(int(x) for x in rel),
            }
        )
    print (f"Loaded {len(out)} questions" )  
    return out


if __name__ == "__main__":
    from app.config import LAW_PATH, TRAIN_PATH

    docs = load_law_documents()
    print(f"Total docs: {len(docs)}")

    # Build a lookup by doc_id
    id2doc = {d["doc_id"]: d for d in docs}

    # 1) Check that sample aid (e.g. 53877) exists and content matches expectations
    sample_aid = 53877
    d = id2doc.get(sample_aid)
    print("\n=== Sample doc 53877 ===")
    if d:
        print("law_id:", d["law_id"])
        print("doc_id:", d["doc_id"])
        print("text snippet:", d["text"][:300].replace("\n", " "))
    else:
        print("NOT FOUND!!")

    # 2) Load *normalised* train and verify gold_doc_ids are valid
    train_data = load_train_data()
    missing = set()
    for q in train_data:
        for aid in q["gold_doc_ids"]:
            if aid not in id2doc:
                missing.add(aid)

    if missing:
        print("\nMISSING gold_doc_ids:", sorted(list(missing))[:50])
    else:
        print("\nAll gold_doc_ids are present in doc_id mapping.")

    # 3) Show one sample question to ensure the structure looks right
    print("\n=== Sample train item ===")
    sample_q = train_data[0]
    print("question_id:", sample_q["question_id"])
    print("text:", sample_q["text"])
    print("gold_doc_ids:", sorted(list(sample_q["gold_doc_ids"]))[:10])
