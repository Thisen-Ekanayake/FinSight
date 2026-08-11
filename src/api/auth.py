# ═══════════════════════════════════════════════════════
# FinSight — Authentication
# ═══════════════════════════════════════════════════════
#
# Purpose : Turn a Google ID token into an email address this deployment is
#           willing to serve, or refuse the request.
#
# Public API:
#   require_user        FastAPI dependency -> the caller's email
#   CurrentUser         Annotated alias for route signatures
#   verify_google_id_token(token) -> claims
#   ANONYMOUS_USER      the identity used when AUTH_ENABLED is false
#   reset_token_cache()
#
# ══ AUTHENTICATION AND AUTHORISATION ARE SEPARATE HERE ══
#   A valid Google token proves who is calling. It says nothing about whether
#   they may call THIS deployment — anyone with a Google account can obtain
#   one. So a token that verifies but whose address is not in
#   AUTH_ALLOWED_EMAILS is a 403, not a 401: signing in again cannot help, and
#   a 401 would send the dashboard into a re-login loop trying.
#
# ══ WHAT THIS MODULE DOES NOT CHECK ══
#   Signature, expiry, audience and issuer are all verify_oauth2_token's job —
#   it validates against Google's published certs and its own issuer list.
#   Re-implementing any of that here would be worse than the library, not
#   safer. What it does NOT do is look at email_verified, so that check is
#   ours: an unverified address must never satisfy an allowlist, or someone
#   who merely claims an address gets in.
#
# ══ WHY SYNC def, NOT async def ══
#   verify_oauth2_token fetches Google's signing certs over the network. On an
#   async dependency that call would block the event loop for every other
#   request in flight. A sync dependency is run by FastAPI in its threadpool,
#   which is the same reasoning that puts run_cycle behind asyncio.to_thread
#   in monitor_routes.py.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core import config as core_config

logger = logging.getLogger(__name__)

# The identity every request carries when AUTH_ENABLED is false. Deliberately
# not an empty string: it shows up in logs as a real value, so "who ran this"
# never reads as a missing field when the answer is "auth was off".
ANONYMOUS_USER = "anonymous@localhost"

# Tolerance for clock drift between whoever minted the token and this host.
# Google's own sample uses the same allowance.
_CLOCK_SKEW_SECONDS = 10

# How long a verified token is trusted without re-checking. Bounded by the
# token's own expiry, so this only ever shortens the window, never extends it.
_CACHE_TTL_SECONDS = 300

# auto_error=False because HTTPBearer's own error is a 403, and a missing
# credential is a 401 — the difference is exactly what the dashboard uses to
# decide between "sign in" and "your account is not permitted". Declaring the
# scheme at all is what puts the Authorize button on /docs.
_bearer = HTTPBearer(auto_error=False, description="Google ID token")

_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_cache_lock = threading.Lock()

_transport: Any = None
_transport_lock = threading.Lock()


def _get_transport() -> Any:
    """
    One shared HTTP transport for cert fetches, built on first use.

    Not built at import time: constructing it imports and opens a requests
    Session, and this module is imported by tooling that never verifies a
    token. Not routed through src/data/cache.py either — that factory exists
    to enforce the SEC User-Agent and per-provider budgets, and an auth check
    has no business inside the data layer's rate accounting.
    """
    global _transport

    with _transport_lock:
        if _transport is None:
            import google.auth.transport.requests

            _transport = google.auth.transport.requests.Request()
        return _transport


def reset_token_cache() -> None:
    """Drop every cached verification. For tests, and for a credential rotation."""
    with _cache_lock:
        _cache.clear()


def _cache_key(token: str) -> str:
    """Hash rather than store the token — a cache dump should not be a credential dump."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cached(key: str) -> dict[str, Any] | None:
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        expires_at, claims = entry
        if expires_at <= now:
            del _cache[key]
            return None
        return claims


def _remember(key: str, claims: dict[str, Any]) -> None:
    """Cache a verified result until the sooner of our TTL and the token's own expiry."""
    now = time.time()
    token_expiry = float(claims.get("exp", 0))
    expires_at = min(now + _CACHE_TTL_SECONDS, token_expiry)
    if expires_at <= now:
        return

    with _cache_lock:
        # Prune while we hold the lock. The cache is keyed by token, so a long
        # -running process would otherwise accumulate one dead entry per hour
        # per signed-in browser and never release them.
        for stale in [k for k, (exp, _) in _cache.items() if exp <= now]:
            del _cache[stale]
        _cache[key] = (expires_at, claims)


def verify_google_id_token(token: str) -> dict[str, Any]:
    """
    Verify a Google ID token and return its claims.

    Signature, expiry, audience and issuer are checked by
    ``verify_oauth2_token`` against Google's published certs. This adds the
    ``email`` and ``email_verified`` checks the library does not make.

    Parameters
    ----------
    token : str
        The raw JWT from the Authorization header.

    Returns
    -------
    dict
        The decoded claims, including ``email``.

    Raises
    ------
    ValueError
        If the token is invalid, expired, for another audience, from another
        issuer, or carries no verified email address.
    """
    cached = _cached(_cache_key(token))
    if cached is not None:
        return cached

    from google.auth import exceptions as google_exceptions
    from google.oauth2 import id_token as google_id_token

    try:
        claims = google_id_token.verify_oauth2_token(
            token,
            _get_transport(),
            audience=core_config.GOOGLE_OAUTH_CLIENT_ID,
            clock_skew_in_seconds=_CLOCK_SKEW_SECONDS,
        )
    except google_exceptions.GoogleAuthError as exc:
        # Raised for a wrong issuer, and it is NOT a ValueError — catching only
        # ValueError would let it escape as a 500.
        raise ValueError(f"token rejected: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"token rejected: {exc}") from exc

    email = str(claims.get("email", "")).strip().lower()
    if not email:
        raise ValueError("token carries no email claim")

    # email_verified can arrive as a bool or as the string "true" depending on
    # how the token was minted; anything else is treated as unverified.
    verified = claims.get("email_verified")
    if verified is not True and str(verified).lower() != "true":
        raise ValueError(f"email {email} is not verified on the Google account")

    claims = {**claims, "email": email}
    _remember(_cache_key(token), claims)
    return claims


def require_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> str:
    """
    Resolve the caller's email, or refuse the request.

    Returns
    -------
    str
        The verified, allowlisted email — or ``ANONYMOUS_USER`` when
        ``AUTH_ENABLED`` is false.

    Raises
    ------
    HTTPException
        401 when no usable credential was presented; 403 when a valid Google
        account is simply not on this deployment's allowlist.
    """
    if not core_config.AUTH_ENABLED:
        return ANONYMOUS_USER

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — send a Google ID token as 'Authorization: Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = verify_google_id_token(credentials.credentials)
    except ValueError as exc:
        logger.info("Rejected a token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            # The reason is deliberately not echoed. It tells an unauthenticated
            # caller how their forgery was spotted and tells a legitimate user
            # nothing they can act on beyond "sign in again".
            detail="Invalid or expired credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    email = str(claims["email"])
    if email not in core_config.AUTH_ALLOWED_EMAILS:
        # 403, not 401. The sign-in worked; retrying it cannot change the
        # outcome, and a 401 would send the dashboard round a re-login loop.
        logger.warning("Refused %s — authenticated but not on AUTH_ALLOWED_EMAILS", email)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{email} is not permitted on this deployment.",
        )

    return email


# For routes that want the caller: `def route(user: CurrentUser) -> ...`.
CurrentUser = Annotated[str, Depends(require_user)]
