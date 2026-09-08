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

# ── Authentication: Google Sign-In ──────────────────────
# OFF by default, and that default is load-bearing. The CLIs, the eval
# harness, and every test construct the app or call the graph without a
# browser anywhere in reach; defaulting to on would make all of them
# unrunnable. Deployments turn it on explicitly — see docs/deploy.md.
AUTH_ENABLED: bool = os.getenv("AUTH_ENABLED", "false").lower() in {"1", "true", "yes"}

# The OAuth 2.0 Web client ID from the Google Cloud console. Public by
# design — it travels in every OAuth flow and GET /auth/config serves it to
# the browser — so it is NOT a secret and does not belong with the keys.
GOOGLE_OAUTH_CLIENT_ID: str = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")


def _parse_emails(raw: str) -> frozenset[str]:
    """Split a comma-separated allowlist, lowercased — addresses are matched case-insensitively."""
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


# Authentication proves WHO is calling; this decides how much they get. These
# addresses are the UNLIMITED tier: they skip the free-query counter entirely.
#
# ══ THIS USED TO BE AN ADMISSION LIST ══
#   It was once the whole authorisation model — a verified account not named
#   here got a 403 and nothing else. It is not that any more. Any verified
#   Google account may now sign in and spend FREE_QUERY_LIMIT queries; naming
#   an address here only exempts it from that meter. Routes that spend
#   unbounded money or disclose configuration still require membership, via
#   require_unlimited_user in src/api/auth.py.
AUTH_ALLOWED_EMAILS: frozenset[str] = _parse_emails(os.getenv("AUTH_ALLOWED_EMAILS", ""))

# How many research queries a signed-in account outside the unlimited tier may
# run, EVER. Lifetime, not per day or per month: there is no reset job and no
# period column, so a counter that has reached the limit stays there until the
# row is edited by hand.
#
# 0 is meaningful and restores the old behaviour — only the unlimited tier can
# query — except the refusal is a 402 carrying a CTA rather than a bare 403.
#
# The limit is NOT frozen onto each row. Raising it later grants the extra
# queries to every existing account; lowering it below what someone has already
# spent leaves them exhausted rather than in debt.
FREE_QUERY_LIMIT: int = int(os.getenv("FREE_QUERY_LIMIT", "5"))

# Where an exhausted account is sent to ask for more. Served in the 402 body
# and by GET /auth/quota rather than hardcoded in the bundle, so a fork or a
# private deployment can point it at its own contact route without a rebuild.
CONTACT_URL: str = os.getenv("CONTACT_URL", "https://github.com/Thisen-Ekanayake/FinSight")

# Extra browser origins allowed to call this API. Empty by default, which means
# same-origin only — the shipped topology serves the bundle and proxies /api
# under one origin (frontend/nginx.conf), so no cross-origin call is ever made.
# Only a deliberately split deployment needs this, and it should name real
# origins: "*" plus a bearer token in the browser lets any page the operator
# visits drive this API.
CORS_ALLOW_ORIGINS: tuple[str, ...] = tuple(
    origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if origin.strip()
)

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

    import google.auth
    import google.auth.exceptions

    try:
        google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as exc:
        raise MissingCredentialError(
            "GEMINI_BACKEND=vertex but no Application Default Credentials found.\n"
            "  Run:  gcloud auth application-default login\n"
            "  Or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key path.\n"
            "  Or run on GCE/Cloud Run/GKE with a service account attached."
        ) from exc


def validate_auth_config() -> None:
    """
    Check that authentication, if switched on, is actually configured.

    Deliberately fatal rather than a warning, for the same reason
    ``QdrantIsolationError`` aborts startup in src/api/main.py: a missing
    client ID means no token can ever be verified, and that should not be
    discovered through a 401 an hour later.

    An empty ``AUTH_ALLOWED_EMAILS`` is NO LONGER fatal. It used to lock every
    account out; now it only means nobody is exempt from the free-query meter,
    which is a legitimate — if rarely intended — configuration. So it warns.

    Does nothing when ``AUTH_ENABLED`` is false, beyond warning if that looks
    like a mistake in production. ``FREE_QUERY_LIMIT`` is checked in both
    modes, because a negative allowance is a typo in any deployment.

    Raises
    ------
    ConfigurationError
        If FREE_QUERY_LIMIT is negative, or AUTH_ENABLED is set with no client ID.
    """
    import logging

    from src.core.errors import ConfigurationError

    if FREE_QUERY_LIMIT < 0:
        raise ConfigurationError(
            f"FREE_QUERY_LIMIT={FREE_QUERY_LIMIT} is negative.\n"
            "  Set it to the number of lifetime queries a free account may run, "
            "or 0 to allow only AUTH_ALLOWED_EMAILS to query."
        )

    if not AUTH_ENABLED:
        if IS_PRODUCTION:
            # Not fatal: a deploy may legitimately sit behind a proxy that
            # authenticates for it (docs/deploy.md §4's basic_auth). Loud,
            # because the other possibility is that nothing is guarding it.
            logging.getLogger(__name__).warning(
                "ENVIRONMENT=production but AUTH_ENABLED=false — the API authenticates nobody. "
                "This is only safe behind a proxy that does (see docs/deploy.md)."
            )
        return

    if not GOOGLE_OAUTH_CLIENT_ID:
        raise ConfigurationError(
            "AUTH_ENABLED=true but GOOGLE_OAUTH_CLIENT_ID is not set.\n"
            "  Create an OAuth 2.0 Web client ID in the Google Cloud console (docs/deploy.md)."
        )

    if not AUTH_ALLOWED_EMAILS:
        # Not fatal any more, but almost never deliberate: it means the
        # operator's own account is metered at FREE_QUERY_LIMIT like everyone
        # else, and that nobody can reach the monitor or admin routes.
        logging.getLogger(__name__).warning(
            "AUTH_ENABLED=true but AUTH_ALLOWED_EMAILS is empty — every account, including "
            "yours, is capped at FREE_QUERY_LIMIT=%d queries, and the monitor and admin "
            "routes are reachable by nobody. Set it to your own address at least.",
            FREE_QUERY_LIMIT,
        )


def ensure_data_dirs() -> None:
    """Create the gitignored runtime directories if they do not yet exist."""
    for path in (DATA_DIR, EDGAR_CACHE_DIR, HTTP_CACHE_PATH.parent, CHECKPOINT_DB.parent):
        path.mkdir(parents=True, exist_ok=True)
