from __future__ import annotations

import argparse

from database.db_embedding import DEFAULT_MODEL_NAME, main as embed_main


def main(model_name: str | None = None, limit: int | None = None) -> None:
    """
    Convenience wrapper to embed chunks with chunk_title prefixed.
    """
    embed_main(
        model_name=model_name,
        limit=limit,
        target="chunks",
        use_chunk_title=True,
        target_column="embedding_with_title_bge_m3",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed article_chunks with chunk_title prefix.")
    parser.add_argument("--model", type=str, default=None, help=f"Embedding model name (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to embed.")
    args = parser.parse_args()
    main(model_name=args.model, limit=args.limit)
