#!/usr/bin/env python3
"""
Export QnA items from MongoDB into the training JSON format.

For each QnA document in the `qna` collection (default DB: `data`), we:
  - Resolve each citation's `legal_document_id` to a law_id via tvpl.processed_documents.diagram.so_hieu
  - Map citation.article_nums (ints) to Postgres articles for that law (source='tvpl')
  - Emit {qid, question, relevant_laws, answer} where relevant_laws are Postgres article doc_ids

Usage (example):
    python scripts/export_qna_from_mongo.py --limit 200 --output data/new_qna_from_mongo.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

from bson import ObjectId
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import execute_values

from app.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    MONGODB_COLLECTION,
    MONGODB_DB,
    MONGODB_URI,
    get_connection,
)


# ----------------------------
# Data models
# ----------------------------

@dataclass
class Citation:
    legal_document_id: str
    article_nums: List[int]


@dataclass
class QnAItem:
    qid: str
    question: str
    answer: str | None
    citations: List[Citation]


# ----------------------------
# Mongo helpers
# ----------------------------

def _to_object_id(raw: str) -> ObjectId:
    try:
        return ObjectId(str(raw))
    except Exception as exc:
        raise ValueError(f"Invalid legal_document_id: {raw}") from exc


def fetch_qna_items(
    client: MongoClient,
    qna_db: str,
    qna_collection: str,
    limit: Optional[int],
    qid_offset: int,
) -> List[QnAItem]:
    coll = client[qna_db][qna_collection]
    cursor = coll.find({}, {"question": 1, "answer": 1, "citations": 1}).sort("_id", -1)
    if limit is not None:
        cursor = cursor.limit(limit)

    out: List[QnAItem] = []
    for idx, doc in enumerate(cursor, start=qid_offset):
        citations: List[Citation] = []
        for cit in doc.get("citations") or []:
            legal_doc = cit.get("legal_document_id")
            if not legal_doc:
                continue
            nums_raw = cit.get("article_nums") or cit.get("article_numbers") or []
            nums: List[int] = []
            for n in nums_raw:
                try:
                    nums.append(int(n))
                except (TypeError, ValueError):
                    continue
            if not nums:
                continue
            citations.append(Citation(legal_document_id=str(legal_doc), article_nums=nums))

        out.append(
            QnAItem(
                # Always assign sequential qid to keep training data stable.
                qid=str(idx),
                question=str(doc.get("question") or "").strip(),
                answer=str(doc.get("answer") or "").strip() or None,
                citations=citations,
            )
        )
    return out


def fetch_law_ids_for_legal_docs(
    client: MongoClient,
    processed_db: str,
    processed_collection: str,
    legal_doc_ids: Iterable[str],
) -> Dict[str, str]:
    """
    Map legal_document_id -> diagram.so_hieu (law_id).
    """
    ids = [_to_object_id(x) for x in legal_doc_ids]
    if not ids:
        return {}

    coll = client[processed_db][processed_collection]
    cursor = coll.find(
        {"_id": {"$in": ids}},
        {"_id": 1, "diagram.so_hieu": 1},
    )
    mapping: Dict[str, str] = {}
    for doc in cursor:
        lid = str(doc.get("_id"))
        law_id = ((doc.get("diagram") or {}).get("so_hieu") or "").strip()
        if lid and law_id:
            mapping[lid] = law_id
    return mapping


def _norm_law_id(val: str) -> str:
    """Lowercase and remove all whitespace for stable comparisons."""
    return "".join((val or "").lower().split())


# ----------------------------
# Postgres helpers
# ----------------------------

def load_articles_for_laws(
    cur,
    law_ids: Sequence[str],
    allow_non_tvpl: bool = False,
) -> Dict[str, List[dict]]:
    """
    Build lookup keyed by normalized law_id -> list of article rows.
    Prefers source='tvpl'; can optionally include other sources for the same law_id.
    """
    if not law_ids:
        return {}

    norm_targets = {_norm_law_id(lid) for lid in law_ids if lid}
    sql = """
        SELECT law_id, article_id, article_idx, doc_id, id, source
        FROM articles
        WHERE regexp_replace(lower(law_id), '\s+', '', 'g') = ANY(%s)
    """
    cur.execute(sql, (list(norm_targets),))
    rows = cur.fetchall()

    lookup: Dict[str, List[dict]] = {}
    for law_id, article_id, article_idx, doc_id, pk, source in rows:
        norm = _norm_law_id(law_id)
        if not allow_non_tvpl and source != "tvpl":
            continue

        entry = {
            "article_id": str(article_id or "").lower(),
            "article_idx": str(article_idx) if article_idx is not None else None,
            "doc_id": doc_id,
            "id": pk,
            "source": source,
        }
        lookup.setdefault(norm, []).append(entry)

    # If mixing sources is allowed, prefer tvpl rows when present.
    if allow_non_tvpl:
        for norm, rows in list(lookup.items()):
            tvpl_rows = [r for r in rows if r["source"] == "tvpl"]
            if tvpl_rows:
                lookup[norm] = tvpl_rows
    return lookup


def _candidate_article_ids(n: int) -> List[str]:
    """
    Strict matches only: dieu_<n> or dieu-<n> (no extra suffix like dieu_1_1).
    Also allow bare numeric article_id (e.g., "12").
    """
    s = str(n)
    return [f"dieu_{s}", f"dieu-{s}", s]


def resolve_doc_ids(
    law_lookup: Dict[str, List[dict]],
    law_id: str,
    article_nums: Iterable[int],
) -> List[int]:
    """
    Map article_nums -> article.doc_id (preferred) else article.id.
    """
    rows = law_lookup.get(_norm_law_id(law_id), [])
    if not rows:
        return []

    resolved: List[int] = []
    for num in article_nums:
        candidates = _candidate_article_ids(num)
        match_doc_id: Optional[int] = None
        match_pk: Optional[int] = None

        for row in rows:
            aid = row["article_id"]
            aidx = row["article_idx"]
            if aid in candidates or (aidx is not None and str(aidx) == str(num)):
                if row["doc_id"] is not None:
                    match_doc_id = row["doc_id"]
                    break
                match_pk = row["id"]

        if match_doc_id is not None:
            resolved.append(int(match_doc_id))
        elif match_pk is not None:
            resolved.append(int(match_pk))

    # Deduplicate while preserving order
    seen: Set[int] = set()
    deduped: List[int] = []
    for x in resolved:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped


# ----------------------------
# Main flow
# ----------------------------

def build_payload(
    qna_items: List[QnAItem],
    law_lookup: Dict[str, List[dict]],
    legal_doc_to_law: Dict[str, str],
) -> List[dict]:
    payload: List[dict] = []
    for item in qna_items:
        rel: List[int] = []
        for cit in item.citations:
            law_id = legal_doc_to_law.get(cit.legal_document_id)
            if not law_id:
                continue
            rel.extend(resolve_doc_ids(law_lookup, law_id, cit.article_nums))

        # Dedup relevant laws per question
        seen: Set[int] = set()
        rel_dedup: List[int] = []
        for x in rel:
            if x in seen:
                continue
            seen.add(x)
            rel_dedup.append(x)

        payload.append(
            {
                "qid": item.qid,
                "question": item.question,
                "answer": item.answer,
                "relevant_laws": rel_dedup,
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export QnA from Mongo to JSON with mapped article doc_ids.")
    parser.add_argument("--mongo-uri", default=MONGODB_URI, help="Mongo connection URI")
    parser.add_argument("--qna-db", default="data", help="Mongo DB containing the qna collection")
    parser.add_argument("--qna-collection", default="qnas", help="Mongo collection with QnA documents")
    parser.add_argument(
        "--processed-db",
        default=MONGODB_DB,
        help="Mongo DB with processed_documents (for law_id lookup, default tvpl)",
    )
    parser.add_argument(
        "--processed-collection",
        default=MONGODB_COLLECTION,
        help="Collection name holding processed_documents",
    )
    parser.add_argument(
        "--qid-offset",
        type=int,
        default=1,
        help="Starting qid for auto-assigned sequential IDs",
    )
    parser.add_argument(
        "--allow-non-tvpl",
        action="store_true",
        help="Allow mapping to articles from any source (fallback if tvpl rows are missing)",
    )
    parser.add_argument(
        "--only-with-relevant",
        action="store_true",
        help="Drop QnA items whose relevant_laws is empty after mapping",
    )
    parser.add_argument(
        "--output-limit",
        type=int,
        default=None,
        help="Maximum number of QnA items to write after filtering/mapping (counts only kept items)",
    )
    parser.add_argument("--output", type=str, default="data/new_qna_from_mongo.json", help="Output JSON path")
    args = parser.parse_args()

    client = MongoClient(args.mongo_uri)
    try:
        qna_items = fetch_qna_items(
            client=client,
            qna_db=args.qna_db,
            qna_collection=args.qna_collection,
            limit=None,
            qid_offset=args.qid_offset,
        )
        print(f"Fetched {len(qna_items)} QnA docs from Mongo ({args.qna_db}.{args.qna_collection}).")
        if not qna_items:
            print("No QnA documents returned. Check --mongo-uri / --qna-db / --qna-collection values.")
            return

        # Collect legal_document_ids and fetch corresponding law_ids (diagram.so_hieu)
        legal_doc_ids: Set[str] = set()
        for item in qna_items:
            for cit in item.citations:
                legal_doc_ids.add(cit.legal_document_id)

        legal_doc_to_law = fetch_law_ids_for_legal_docs(
            client=client,
            processed_db=args.processed_db,
            processed_collection=args.processed_collection,
            legal_doc_ids=legal_doc_ids,
        )
        print(
            f"Resolved {len(legal_doc_to_law)}/{len(legal_doc_ids)} legal_document_ids "
            f"to law_id via {args.processed_db}.{args.processed_collection}."
        )

        # Build article lookup from Postgres for all involved laws
        conn = get_connection()
        try:
            cur = conn.cursor()
            law_lookup = load_articles_for_laws(cur, list(set(legal_doc_to_law.values())))
        finally:
            conn.close()
        print(f"Loaded article lookup for {len(law_lookup)} laws from Postgres (source='tvpl').")
        missing_laws = set(legal_doc_to_law.values()) - set(law_lookup.keys())
        if missing_laws:
            missing_samples = sorted(list(missing_laws))[:10]
            print(
                f"WARNING: {len(missing_laws)} law_id values referenced by QnA are not in Postgres "
                f"(source='tvpl'). Examples: {missing_samples}"
            )

        payload = build_payload(qna_items, law_lookup, legal_doc_to_law)
        zero_rel = sum(1 for item in payload if not item.get("relevant_laws"))
        if zero_rel:
            print(f"WARNING: {zero_rel} QnA items have empty relevant_laws after mapping.")

        if args.only_with_relevant:
            before = len(payload)
            payload = [p for p in payload if p.get("relevant_laws")]
            print(f"Filtered out {before - len(payload)} items with empty relevant_laws; kept {len(payload)}.")

        if args.output_limit is not None:
            payload = payload[: args.output_limit]
            print(f"Truncated output to {len(payload)} items (requested limit={args.output_limit}).")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Wrote {len(payload)} QnA items to {args.output}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
