#!/usr/bin/env python3
"""
Extract articles (điều) from processed_documents in MongoDB.

This script takes a document _id and extracts all articles with their
metadata including: chapter, section, clauses, document title, etc.
Requires `MONGODB_URI` to be set in the environment (loaded from .env if present).

Usage:
    python scripts/extract_articles.py <document_id>

Example:
    python scripts/extract_articles.py 693300a913b0444ae980c268
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient


# MongoDB configuration
DATABASE_NAME = "tvpl"
COLLECTION_NAME = "processed_documents"


@dataclass
class Citation:
    """Represents a citation/reference in a clause."""

    part_link_id: str
    content: str
    tvpl_id: str
    tvpl_link: str
    marked_content: str
    marked_content_index_from: int
    marked_content_index_to: int
    citation_type: str  # 'type' is reserved in Python
    legal_document_id: str | None


@dataclass
class Clause:
    """Represents a clause (khoản) within an article."""

    id: str
    content: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Article:
    """Represents an article (điều) with full metadata."""

    id: str
    article_number: int | None
    content: str
    clauses: list[Clause] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

    # Hierarchical metadata
    chapter_id: str | None = None
    chapter_name: str | None = None
    section_id: str | None = None  # mục
    section_name: str | None = None

    # Document metadata
    document_id: str | None = None
    document_title: str | None = None
    document_number: str | None = None  # số hiệu
    document_type: str | None = None  # loại văn bản
    issuing_authority: str | None = None  # nơi ban hành
    issue_date: str | None = None  # ngày ban hành
    effective_date: str | None = None  # ngày hiệu lực


@dataclass
class ExtractionResult:
    """Result of article extraction from a document."""

    document_id: str
    document_title: str
    document_metadata: dict[str, Any]
    articles: list[Article]
    total_articles: int
    total_clauses: int


def get_mongodb_client() -> MongoClient:
    """Get MongoDB client using MONGODB_URI from env/.env."""
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set in the environment or .env file")
    return MongoClient(uri)


def parse_citation(cit_data: dict[str, Any] | None) -> Citation | None:
    """Parse a citation from MongoDB document."""
    if not cit_data:
        return None

    return Citation(
        part_link_id=cit_data.get("part_link_id", ""),
        content=cit_data.get("content", ""),
        tvpl_id=cit_data.get("tvpl_id", ""),
        tvpl_link=cit_data.get("tvpl_link", ""),
        marked_content=cit_data.get("marked_content", ""),
        marked_content_index_from=cit_data.get("marked_content_index_from", 0),
        marked_content_index_to=cit_data.get("marked_content_index_to", 0),
        citation_type=cit_data.get("type", ""),
        legal_document_id=cit_data.get("legal_document_id"),
    )


def extract_article_number(article_id: str) -> int | None:
    """Extract article number from article ID (e.g., 'dieu_1' -> 1)."""
    match = re.search(r"dieu_(\d+)", article_id)
    return int(match.group(1)) if match else None


def build_toc_hierarchy(table_of_content: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build a mapping of article IDs to their chapter/section hierarchy from TOC.

    Returns a dict like:
    {
        "dieu_1": {
            "chapter_id": "chuong_1",
            "chapter_name": "Chương I NHỮNG QUY ĐỊNH CHUNG",
            "section_id": None,
            "section_name": None
        },
        "dieu_14": {
            "chapter_id": "chuong_3",
            "chapter_name": "Chương III AN TOÀN BỨC XẠ...",
            "section_id": "muc_1_3",
            "section_name": "Mục 1. QUY ĐỊNH CHUNG"
        }
    }
    """
    hierarchy = {}

    def traverse(
        items: list[dict[str, Any]],
        current_chapter_id: str | None = None,
        current_chapter_name: str | None = None,
        current_section_id: str | None = None,
        current_section_name: str | None = None,
    ):
        for item in items:
            key = item.get("key", "").lstrip("#")
            label = item.get("label", "")
            children = item.get("children", [])

            # Detect chapter
            if key.startswith("chuong_") and not key.endswith("_name"):
                current_chapter_id = key
                current_chapter_name = label
                current_section_id = None
                current_section_name = None

            # Detect section (mục)
            elif key.startswith("muc_"):
                current_section_id = key
                current_section_name = label

            # Detect article (điều)
            elif key.startswith("dieu_"):
                hierarchy[key] = {
                    "chapter_id": current_chapter_id,
                    "chapter_name": current_chapter_name,
                    "section_id": current_section_id,
                    "section_name": current_section_name,
                }

            # Recursively process children
            if children:
                traverse(
                    children,
                    current_chapter_id,
                    current_chapter_name,
                    current_section_id,
                    current_section_name,
                )

    items = table_of_content.get("items", [])
    traverse(items)

    return hierarchy


def extract_articles_from_document(doc: dict[str, Any]) -> ExtractionResult:
    """Extract all articles with metadata from a processed document."""
    doc_id = str(doc.get("_id", ""))

    # Extract document metadata from diagram
    diagram = doc.get("diagram", {})
    doc_title = diagram.get("ten", "")
    doc_number = diagram.get("so_hieu", "")
    doc_type = diagram.get("loai_van_ban", "")
    issuing_authority = diagram.get("noi_ban_hanh", "")
    issue_date = diagram.get("ngay_ban_hanh", "")
    effective_date = diagram.get("ngay_hieu_luc", "")

    document_metadata = {
        "title": doc_title,
        "number": doc_number,
        "type": doc_type,
        "field": diagram.get("linh_vuc_nganh", ""),
        "issuing_authority": issuing_authority,
        "signer": diagram.get("nguoi_ky", ""),
        "issue_date": issue_date,
        "effective_date": effective_date,
        "publish_date": diagram.get("ngay_dang", ""),
        "gazette_number": diagram.get("so_cong_bao", ""),
        "status": diagram.get("tinh_trang", ""),
        "url": doc.get("url", ""),
        "law_id": doc.get("law_id", ""),
    }

    # Build hierarchy from table of content
    table_of_content = doc.get("table_of_content", {})
    toc_hierarchy = build_toc_hierarchy(table_of_content)

    # Parse content_parts to build articles with clauses
    content_parts = doc.get("content_parts", [])

    # Build a map of content parts by ID
    parts_map: dict[str, dict[str, Any]] = {}
    for part in content_parts:
        part_id = part.get("id", "")
        if part_id:
            parts_map[part_id] = part

    # Find all articles and their clauses
    articles: list[Article] = []
    article_parts = [p for p in content_parts if p.get("level") == "article"]

    for article_part in article_parts:
        article_id = article_part.get("id", "")

        # Get hierarchy info from TOC
        hierarchy_info = toc_hierarchy.get(article_id, {})

        # Parse citations for the article
        article_citations = []
        if article_part.get("citations"):
            for cit in article_part["citations"]:
                parsed_cit = parse_citation(cit)
                if parsed_cit:
                    article_citations.append(parsed_cit)

        # Find all clauses belonging to this article
        clauses: list[Clause] = []
        for part in content_parts:
            if part.get("level") == "clause" and part.get("parent_id") == article_id:
                clause_citations = []
                if part.get("citations"):
                    for cit in part["citations"]:
                        parsed_cit = parse_citation(cit)
                        if parsed_cit:
                            clause_citations.append(parsed_cit)

                clause = Clause(
                    id=part.get("id", ""),
                    content=part.get("content", ""),
                    citations=clause_citations,
                )
                clauses.append(clause)

        # Sort clauses by their ID to maintain order
        clauses.sort(key=lambda c: _extract_clause_order(c.id))

        article = Article(
            id=article_id,
            article_number=extract_article_number(article_id),
            content=article_part.get("content", ""),
            clauses=clauses,
            citations=article_citations,
            chapter_id=hierarchy_info.get("chapter_id"),
            chapter_name=hierarchy_info.get("chapter_name"),
            section_id=hierarchy_info.get("section_id"),
            section_name=hierarchy_info.get("section_name"),
            document_id=doc_id,
            document_title=doc_title,
            document_number=doc_number,
            document_type=doc_type,
            issuing_authority=issuing_authority,
            issue_date=issue_date,
            effective_date=effective_date,
        )
        articles.append(article)

    # Sort articles by their number
    articles.sort(key=lambda a: a.article_number or 0)

    total_clauses = sum(len(a.clauses) for a in articles)

    return ExtractionResult(
        document_id=doc_id,
        document_title=doc_title,
        document_metadata=document_metadata,
        articles=articles,
        total_articles=len(articles),
        total_clauses=total_clauses,
    )


def _extract_clause_order(clause_id: str) -> tuple[int, int]:
    """Extract ordering info from clause ID (e.g., 'khoan_1_dieu_1' -> (1, 1))."""
    match = re.search(r"khoan_(\d+)_dieu_(\d+)", clause_id)
    if match:
        return (int(match.group(2)), int(match.group(1)))
    return (0, 0)


def get_article_full_content(article: Article) -> str:
    """Get the full content of an article including all clauses."""
    parts = [article.content] if article.content else []

    for clause in article.clauses:
        if clause.content:
            parts.append(clause.content)

    return "\n\n".join(parts)


def print_article_summary(result: ExtractionResult):
    """Print a summary of extracted articles."""
    print(f"\n{'=' * 80}")
    print(f"Document: {result.document_title}")
    print(f"Document ID: {result.document_id}")
    print(f"Total Articles: {result.total_articles}")
    print(f"Total Clauses: {result.total_clauses}")
    print(f"{'=' * 80}\n")

    current_chapter = None
    current_section = None

    for article in result.articles:
        # Print chapter header if changed
        if article.chapter_name and article.chapter_name != current_chapter:
            current_chapter = article.chapter_name
            current_section = None  # Reset section when chapter changes
            print(f"\n{'─' * 80}")
            print(f"📚 {current_chapter}")
            print(f"{'─' * 80}")

        # Print section header if changed
        if article.section_name and article.section_name != current_section:
            current_section = article.section_name
            print(f"\n  📑 {current_section}")

        # Print article
        content_first_line = article.content.split("\n")[0] if article.content else ""
        print(f"\n    📄 {content_first_line}")
        print(f"       ID: {article.id} | Clauses: {len(article.clauses)}")

        # Print clauses summary
        for clause in article.clauses[:3]:  # Show first 3 clauses
            content_preview = (
                clause.content[:100] + "..."
                if len(clause.content) > 100
                else clause.content
            )
            print(f"       └─ {content_preview}")

        if len(article.clauses) > 3:
            print(f"       └─ ... and {len(article.clauses) - 3} more clauses")


def main():
    parser = argparse.ArgumentParser(
        description="Extract articles from processed_documents in MongoDB"
    )
    parser.add_argument(
        "document_id",
        type=str,
        help="The _id of the document to extract articles from",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for JSON result (optional)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary of extracted articles",
    )

    args = parser.parse_args()

    # Connect to MongoDB
    print("Connecting to MongoDB...")
    try:
        client = get_mongodb_client()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    # Fetch the document
    try:
        doc_id = ObjectId(args.document_id)
    except Exception:
        print(f"Error: Invalid ObjectId format: {args.document_id}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching document: {args.document_id}")
    doc = collection.find_one({"_id": doc_id})

    if not doc:
        print(f"Error: Document not found: {args.document_id}", file=sys.stderr)
        sys.exit(1)

    # Extract articles
    print("Extracting articles...")
    result = extract_articles_from_document(doc)

    # Print summary if requested
    if args.summary:
        print_article_summary(result)

    # Prepare output
    output_data = {
        "document_id": result.document_id,
        "document_title": result.document_title,
        "document_metadata": result.document_metadata,
        "total_articles": result.total_articles,
        "total_clauses": result.total_clauses,
        "articles": [],
    }

    for article in result.articles:
        article_data = {
            "id": article.id,
            "article_number": article.article_number,
            "content": article.content,
            "chapter_id": article.chapter_id,
            "chapter_name": article.chapter_name,
            "section_id": article.section_id,
            "section_name": article.section_name,
            "clauses_count": len(article.clauses),
            "clauses": [
                {
                    "id": c.id,
                    "content": c.content,
                    "citations": [asdict(cit) for cit in c.citations],
                }
                for c in article.clauses
            ],
            "citations": [asdict(cit) for cit in article.citations],
        }

        # Always include full_content
        article_data["full_content"] = get_article_full_content(article)

        output_data["articles"].append(article_data)

    # Output result
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nOutput saved to: {args.output}")
    else:
        print(json.dumps(output_data, ensure_ascii=False, indent=2))

    print(
        f"\n✅ Extracted {result.total_articles} articles with {result.total_clauses} clauses"
    )

    client.close()


if __name__ == "__main__":
    main()
