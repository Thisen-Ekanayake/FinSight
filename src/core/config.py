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
# Two ways to reach Gemini, selected by GEMINI_BACKEND:
#
#   "vertex"   -> Vertex AI via Application Default Credentials (gcloud auth
#                 application-default login). Needs a GCP project with
#                 aiplatform.googleapis.com enabled. BILLED PER TOKEN.
#   "aistudio" -> Google AI Studio via a GOOGLE_API_KEY. Free tier, but
#                 metered in requests-per-minute.
#
# Both go through get_llm() in src/core/llm.py, so nothing downstream cares.
GEMINI_BACKEND: str = os.getenv("GEMINI_BACKEND", "vertex").lower()

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Vertex AI targeting. GOOGLE_CLOUD_PROJECT is the conventional name and is
# what the google-auth libraries themselves look for.
GCP_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "") or os.getenv("GCP_PROJECT", "")
GCP_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "") or os.getenv("GCP_LOCATION", "us-central1")

ModelTier = Literal["flash", "pro"]

GEMINI_MODEL_FLASH: str = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")
GEMINI_MODEL_PRO: str = os.getenv("GEMINI_MODEL_PRO", "gemini-2.5-pro")
GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))
GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "5"))

# Drives the per-tier rate limiters in src/core/llm.py.
#
# On "aistudio" this is a QUOTA guard: the free tier meters requests-per-minute
# and exceeding it surfaces as 429s. Set to ~80% of your real quota
# (https://ai.google.dev/gemini-api/docs/rate-limits).
#
# On "vertex" it is a COST guard instead: Vertex bills per token with no free
# tier, and quota is high enough that nothing stops a runaway fan-out from
# spending real money. The limiter is the throttle. Raise deliberately.
GEMINI_RPM: dict[str, int] = {
    "flash": int(os.getenv("GEMINI_RPM_FLASH", "30")),
    "pro": int(os.getenv("GEMINI_RPM_PRO", "10")),
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


def validate_llm_credentials() -> None:
    """
    Check that whichever Gemini backend is selected can actually authenticate.

    Fails fast with an actionable message rather than letting a 401 surface
    from deep inside a graph run.

    Raises
    ------
    ConfigurationError
        If GEMINI_BACKEND is not a known value.
    MissingCredentialError
        If the selected backend's credentials are absent.
    """
    from src.core.errors import ConfigurationError, MissingCredentialError

    if GEMINI_BACKEND == "aistudio":
        require_key("GOOGLE_API_KEY")
        return

    if GEMINI_BACKEND != "vertex":
        raise ConfigurationError(f"GEMINI_BACKEND must be 'vertex' or 'aistudio', got {GEMINI_BACKEND!r}")

    if not GCP_PROJECT:
        raise MissingCredentialError(
            "GEMINI_BACKEND=vertex but no GCP project is set.\n"
            "  Set GOOGLE_CLOUD_PROJECT in .env (see docs/api_keys.md)."
        )

    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    explicit = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not explicit and not adc_path.is_file():
        raise MissingCredentialError(
            "GEMINI_BACKEND=vertex but no Application Default Credentials found.\n"
            "  Run:  gcloud auth application-default login\n"
            "  Or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key path."
        )


def ensure_data_dirs() -> None:
    """Create the gitignored runtime directories if they do not yet exist."""
    for path in (DATA_DIR, EDGAR_CACHE_DIR, HTTP_CACHE_PATH.parent, CHECKPOINT_DB.parent):
        path.mkdir(parents=True, exist_ok=True)
