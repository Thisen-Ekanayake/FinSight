# ═══════════════════════════════════════════════════════
# FinSight — Core Configuration
# ═══════════════════════════════════════════════════════
#
# Purpose : Single source of truth for environment-derived settings.
#           Every other module imports constants from here rather than
#           reading os.environ directly.
#
# Public API:
#   GOOGLE_API_KEY, GEMINI_MODEL_FLASH, GEMINI_MODEL_PRO, ...
#   QDRANT_URL, EMBEDDING_MODEL, EMBEDDING_DIM
#   LANGSMITH_* , DATABASE_URL, CHECKPOINT_DB
#   ModelTier, require_key()
#
# Usage:
#   from src.core.config import GEMINI_MODEL_FLASH, require_key
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass


# ── Paths ───────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative configured path against the project root."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ── LLM: Google Gemini ──────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

ModelTier = Literal["flash", "pro"]

GEMINI_MODEL_FLASH: str = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")
GEMINI_MODEL_PRO: str = os.getenv("GEMINI_MODEL_PRO", "gemini-2.5-pro")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "5"))

# Free-tier Gemini limits requests-per-minute, not spend. These drive the
# per-tier rate limiters in src/core/llm.py. Set them to ~80% of your actual
# quota (https://ai.google.dev/gemini-api/docs/rate-limits) so bursts have
# headroom. Exceeding the quota surfaces as 429s, not as a bill.
GEMINI_RPM: dict[str, int] = {
    "flash": int(os.getenv("GEMINI_RPM_FLASH", "10")),
    "pro": int(os.getenv("GEMINI_RPM_PRO", "4")),
}

MODEL_BY_TIER: dict[str, str] = {
    "flash": GEMINI_MODEL_FLASH,
    "pro": GEMINI_MODEL_PRO,
}

# ── Observability: LangSmith ────────────────────────────
LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes"}
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "finsight-dev")
LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# ── Vector store: Qdrant ────────────────────────────────
# NOTE: 6335, not 6333. See src/vectorstore/client.py for why.
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6335")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_TIMEOUT: int = int(os.getenv("QDRANT_TIMEOUT", "30"))

# ── Embeddings ──────────────────────────────────────────
EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "fastembed")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))

# ── Persistence ─────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/finsight.db")
CHECKPOINT_DB: Path = _resolve(os.getenv("CHECKPOINT_DB", "data/checkpoints.sqlite"))
HTTP_CACHE_PATH: Path = _resolve(os.getenv("HTTP_CACHE_PATH", "data/cache/http_cache"))
EDGAR_CACHE_DIR: Path = _resolve(os.getenv("EDGAR_CACHE_DIR", "data/edgar"))

# ── Runtime ─────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION: bool = ENVIRONMENT.lower() == "production"


# ── Validation helpers ──────────────────────────────────
def require_key(name: str) -> str:
    """
    Fetch a required environment variable, failing loudly if it is missing.

    Deferred rather than validated at import time so that unit tests and
    offline tooling can import this module without any credentials present.

    Parameters
    ----------
    name : str
        Environment variable name, e.g. ``"GOOGLE_API_KEY"``.

    Returns
    -------
    str
        The variable's value.

    Raises
    ------
    MissingCredentialError
        If the variable is unset or empty.
    """
    from src.core.errors import MissingCredentialError

    value = os.getenv(name, "")
    if not value:
        raise MissingCredentialError(
            f"{name} is not set. Copy .env.example to .env and fill it in "
            f"(see docs/api_keys.md for where to get each key)."
        )
    return value


def ensure_data_dirs() -> None:
    """Create the gitignored runtime directories if they do not yet exist."""
    for path in (DATA_DIR, EDGAR_CACHE_DIR, HTTP_CACHE_PATH.parent, CHECKPOINT_DB.parent):
        path.mkdir(parents=True, exist_ok=True)
