# ═══════════════════════════════════════════════════════
# FinSight — HTTP Caching & Session Factory
# ═══════════════════════════════════════════════════════
#
# Purpose : Every outbound HTTP call in FinSight goes through a session built
#           here. That single choke point is what makes it impossible to call
#           the SEC without a valid User-Agent.
#
# Public API:
#   get_session(provider)       cached, retrying, correctly-headered session
#   fetch_json(provider, url)   guarded GET returning parsed JSON
#   fetch_text(provider, url)   guarded GET returning text
#   cached_file(...)            disk cache for immutable documents
#   clear_http_cache()
#
# Two tiers of caching:
#   1. requests-cache (SQLite) for API responses, TTL per CACHE_TTL.
#   2. Plain disk files for EDGAR filing documents, which are immutable once
#      accepted — an accession number will never point at different bytes, so
#      they are cached forever and re-ingest costs zero live requests.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import requests

from src.core.errors import DataSourceError
from src.data.config import (
    CACHE_TTL,
    DEFAULT_CACHE_TTL,
    HTTP_BACKOFF_FACTOR,
    HTTP_CACHE_PATH,
    HTTP_MAX_RETRIES,
    HTTP_RETRY_STATUSES,
    HTTP_TIMEOUT,
    validate_sec_user_agent,
)
from src.data.rate_limit import guard, record_call

logger = logging.getLogger(__name__)

_SESSIONS: dict[str, requests.Session] = {}
_SESSION_LOCK = threading.Lock()

# Providers whose requests must carry the SEC contact header.
_SEC_PROVIDERS: frozenset[str] = frozenset({"sec", "edgar", "edgar_xbrl"})


def _build_session(provider: str) -> requests.Session:
    """Construct a caching, retrying session with provider-appropriate headers."""
    import requests_cache
    from urllib3.util.retry import Retry

    HTTP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    session = requests_cache.CachedSession(
        cache_name=str(HTTP_CACHE_PATH),
        backend="sqlite",
        expire_after=DEFAULT_CACHE_TTL,
        allowable_codes=(200,),
        stale_if_error=True,  # a cached body beats a hard failure
    )

    retry = Retry(
        total=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF_FACTOR,
        status_forcelist=list(HTTP_RETRY_STATUSES),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    if provider in _SEC_PROVIDERS:
        # Validated here, at the only place a SEC session can be created, so a
        # placeholder address cannot reach the SEC by any code path.
        session.headers.update(
            {
                "User-Agent": validate_sec_user_agent(),
                "Accept-Encoding": "gzip, deflate",
            }
        )
    else:
        session.headers.update({"User-Agent": "FinSight/0.1"})

    return session


def get_session(provider: str) -> requests.Session:
    """
    Return the shared session for a provider, building it on first use.

    Parameters
    ----------
    provider : str
        Provider key, e.g. ``"sec"``, ``"fred"``, ``"finnhub"``.

    Returns
    -------
    requests.Session
        Cached and retrying, with the provider's required headers applied.
    """
    with _SESSION_LOCK:
        if provider not in _SESSIONS:
            _SESSIONS[provider] = _build_session(provider)
            logger.debug("Built HTTP session for %s", provider)
        return _SESSIONS[provider]


def _request(
    provider: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    ttl_key: str | None = None,
) -> requests.Response:
    """Guarded GET: budget check, rate limit, fetch, raise on failure."""
    guard(provider)

    session = get_session(provider)
    expire = CACHE_TTL.get(ttl_key or "", DEFAULT_CACHE_TTL) if ttl_key else DEFAULT_CACHE_TTL
    # requests-cache uses -1 to mean "never expire".
    expire_after = -1 if expire is None else expire

    try:
        response = session.get(url, params=params, timeout=HTTP_TIMEOUT, expire_after=expire_after)
    except requests.RequestException as exc:
        raise DataSourceError(provider, f"request to {url} failed: {exc}") from exc

    # Only a live call counts against the daily budget; cache hits are free.
    if not getattr(response, "from_cache", False):
        record_call(provider)

    if response.status_code == 403 and provider in _SEC_PROVIDERS:
        raise DataSourceError(
            provider,
            "SEC returned 403 — check SEC_USER_AGENT carries a real contact email and you are under 10 req/s.",
        )
    if not response.ok:
        raise DataSourceError(provider, f"HTTP {response.status_code} from {url}: {response.text[:200]}")

    return response


def fetch_json(
    provider: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    ttl_key: str | None = None,
) -> Any:
    """
    GET a URL and parse JSON, respecting rate limits, budgets, and cache.

    Parameters
    ----------
    provider : str
        Provider key for rate limiting and budgeting.
    url : str
        Absolute URL.
    params : dict, optional
        Query parameters.
    ttl_key : str, optional
        Key into CACHE_TTL selecting this response's lifetime.

    Returns
    -------
    Any
        Parsed JSON.

    Raises
    ------
    DataSourceError
        On transport failure, non-2xx status, or unparseable body.
    """
    response = _request(provider, url, params=params, ttl_key=ttl_key)
    try:
        return response.json()
    except ValueError as exc:
        raise DataSourceError(provider, f"invalid JSON from {url}: {exc}") from exc


def fetch_text(
    provider: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    ttl_key: str | None = None,
) -> str:
    """GET a URL and return its body as text. See fetch_json for semantics."""
    return _request(provider, url, params=params, ttl_key=ttl_key).text


def cached_file(
    provider: str,
    url: str,
    dest: Path,
    *,
    ttl_key: str | None = None,
) -> Path:
    """
    Download a URL to disk once and reuse it forever.

    For EDGAR filing documents: a filing is immutable once accepted, so its
    accession number will never point at different bytes. Caching to disk
    means re-ingest and reproducibility cost zero live requests.

    Parameters
    ----------
    provider : str
        Provider key for rate limiting.
    url : str
        Document URL.
    dest : Path
        Destination path; parent directories are created.
    ttl_key : str, optional
        Unused for immutable documents; accepted for signature symmetry.

    Returns
    -------
    Path
        ``dest``, guaranteed to exist.

    Raises
    ------
    DataSourceError
        If the download fails and no cached copy exists.
    """
    if dest.is_file() and dest.stat().st_size > 0:
        logger.debug("Disk cache hit: %s", dest.name)
        return dest

    guard(provider)
    session = get_session(provider)

    try:
        with session.get(url, timeout=HTTP_TIMEOUT, stream=True) as response:
            if not response.ok:
                raise DataSourceError(provider, f"HTTP {response.status_code} downloading {url}")

            dest.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file first so an interrupted download never
            # leaves a truncated file that later looks like a cache hit.
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=65536):
                    fh.write(chunk)
            tmp.replace(dest)
    except requests.RequestException as exc:
        raise DataSourceError(provider, f"download of {url} failed: {exc}") from exc

    record_call(provider)
    logger.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


def clear_http_cache() -> None:
    """Drop cached sessions and the on-disk HTTP cache. Does not touch EDGAR files."""
    with _SESSION_LOCK:
        for session in _SESSIONS.values():
            cache = getattr(session, "cache", None)
            if cache is not None:
                cache.clear()
            session.close()
        _SESSIONS.clear()
