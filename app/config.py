from dotenv import load_dotenv
import os
from pathlib import Path

import psycopg2
from openai import OpenAI

load_dotenv()

# ========== OpenAI / OpenRouter ==========
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

EMB_MODEL = os.getenv("EMB_MODEL", "openai/text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash-lite-preview-09-2025")

# ========== Paths ==========
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

LAW_PATH = Path(os.getenv("LAW_PATH", DATA_DIR / "legal_corpus.json"))
TRAIN_PATH = Path(os.getenv("TRAIN_PATH", DATA_DIR / "train.json"))

STOPWORDS_PATH = Path(os.getenv("STOPWORDS_PATH", DATA_DIR / "vietnamese-stopwords.txt"))

# ========== Embedding settings ==========
MAX_CHARS = int(os.getenv("MAX_CHARS", "4000"))          # max chars per chunk sent to embeddings API
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))          # texts per API request
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "50"))          # commit every N batches/ops (used by various scripts)

# Concurrency for embedding script (multiple API calls at once)
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "16"))

# Chunking settings
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", str(MAX_CHARS)))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "200"))  # prevent tiny chunks when possible

# ========== DB config (Postgres / pgvector) ==========
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "legal_qa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "secret")
DB_NAME = os.getenv("DB_NAME", "legal_qa")


JINA_API_KEY = os.getenv("JINA_API_KEY")
def get_connection():
    """Return a psycopg2 connection to the Postgres DB."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )


def get_client() -> OpenAI:
    """Return an OpenAI client (used for embeddings + QA)."""
    return OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )
