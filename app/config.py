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
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")

# ========== Paths ==========
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Law corpora (adjust these in .env to match your real files)
LAW_PATH = Path(os.getenv("LAW_PATH", DATA_DIR / "legal_corpus.json"))
TRAIN_PATH = Path(os.getenv("TRAIN_PATH", DATA_DIR / "train.json"))

# ARTICLE_EMB_PATH = Path(os.getenv("ARTICLE_EMB_PATH", DATA_DIR / "article_embeddings.npy"))
# TRAIN_Q_EMB_PATH = Path(os.getenv("TRAIN_Q_EMB_PATH", DATA_DIR / "train_question_embeddings.npy"))
# TEST_Q_EMB_PATH = Path(os.getenv("TEST_Q_EMB_PATH", DATA_DIR / "test_question_embeddings.npy"))

# ARTICLE_TOKENS_PATH = Path(os.getenv("ARTICLE_TOKENS_PATH", DATA_DIR / "article_tokens.json"))
# STOPWORDS_PATH = Path(os.getenv("STOPWORDS_PATH", DATA_DIR / "vietnamese-stopwords.txt"))

# ========== Embedding settings ==========
MAX_CHARS = int(os.getenv("MAX_CHARS", "4000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "50"))

# ========== DB config (Postgres / pgvector) ==========
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = int(os.getenv("DB_PORT", "5400"))
DB_USER = os.getenv("DB_USER", "legal_qa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "secret")
DB_NAME = os.getenv("DB_NAME", "legal_qa")


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
