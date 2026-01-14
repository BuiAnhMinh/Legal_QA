from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from database.db_embedding import DEFAULT_MODEL_NAME, main as embed_main


def main(
    model_name: str | None = None,
    limit: int | None = None,
    price_per_1k: float | None = None,
) -> None:
    """
    Convenience wrapper to embed chunks with chunk_title prefixed.
    """
    embed_main(
        model_name=model_name,
        limit=limit,
        target="chunks",
        use_chunk_title=True,
        target_column="embedding_with_title_bge_m3",
        price_per_1k=price_per_1k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed article_chunks with chunk_title prefix.")
    parser.add_argument("--model", type=str, default=None, help=f"Embedding model name (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to embed.")
    parser.add_argument(
        "--price-per-1k",
        type=float,
        default=None,
        help="Optional price per 1K tokens to estimate embedding cost.",
    )
    args = parser.parse_args()
    main(model_name=args.model, limit=args.limit, price_per_1k=args.price_per_1k)
