# ═══════════════════════════════════════════════════════
# FinSight — Tests: Authentication
# ═══════════════════════════════════════════════════════
#
# Nothing here reaches Google. verify_oauth2_token is replaced, so these test
# OUR half — the allowlist, the 401/403 split, the email_verified rule, the
# cache — and not the library's signature checking, which is not ours to
# re-test.
#
# ══ THE TEST THAT MATTERS MOST ══
#   test_health_is_reachable_without_a_token. /health behind auth makes the api
#   container permanently unhealthy, so the web container's depends_on never
#   releases and the dashboard never starts. That failure looks like a broken
#   frontend and would be debugged nowhere near this file.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api import auth
from src.api.auth_routes import router as auth_router
from src.core import config as core_config
from src.core.errors import ConfigurationError

ALLOWED = "owner@example.com"
CLIENT_ID = "cid.apps.googleusercontent.com"


def _claims(**overrides: Any) -> dict[str, Any]:
    """A decoded token as verify_oauth2_token would return it."""
    return {
        "email": ALLOWED,
        "email_verified": True,
        "aud": CLIENT_ID,
        "iss": "https://accounts.google.com",
        "exp": time.time() + 3600,
        **overrides,
    }


def _returns(claims: dict[str, Any]) -> Any:
    """A stand-in for verify_oauth2_token that accepts any token and returns `claims`."""

    def fake(token: Any, request: Any, audience: Any = None, clock_skew_in_seconds: int = 0) -> dict[str, Any]:
        return claims

    return fake


@pytest.fixture(autouse=True)
def auth_on() -> Iterator[None]:
    """
    Auth switched on, with a one-address allowlist.

    Autouse and always reset: the token cache is module state, so a verified
    result leaking between tests would let a later assertion pass against an
    earlier test's token.
    """
    auth.reset_token_cache()
    with (
        patch.object(core_config, "AUTH_ENABLED", True),
        patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", CLIENT_ID),
        patch.object(core_config, "AUTH_ALLOWED_EMAILS", frozenset({ALLOWED})),
        # Building the real transport opens a requests Session that no test needs.
        patch.object(auth, "_get_transport", lambda: object()),
    ):
        yield
    auth.reset_token_cache()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A minimal app: one guarded route that echoes the caller, plus /auth/config."""
    app = FastAPI()

    @app.get("/protected")
    def protected(user: str = Depends(auth.require_user)) -> dict[str, str]:
        return {"user": user}

    app.include_router(auth_router)
    with TestClient(app) as test_client:
        yield test_client


def _get(client: TestClient, token: str | None = None, path: str = "/protected") -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.get(path, headers=headers)


# ── The happy path ──────────────────────────────────────
def test_allowlisted_account_is_admitted_and_identified(client: TestClient) -> None:
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims())):
        response = _get(client, "good-token")

    assert response.status_code == 200
    # Not just 200 — the route must receive the caller, since that identity is
    # the whole reason for preferring this over a shared password.
    assert response.json() == {"user": ALLOWED}


def test_email_is_matched_case_insensitively(client: TestClient) -> None:
    """Google may return the address in any case; the allowlist is lowercased."""
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email="Owner@Example.COM"))):
        response = _get(client, "mixed-case")

    assert response.status_code == 200
    assert response.json() == {"user": ALLOWED}


# ── Authentication failures are 401 ─────────────────────
def test_missing_header_is_401_with_a_challenge(client: TestClient) -> None:
    response = _get(client)
    assert response.status_code == 401
    # Without this header the response is not a well-formed 401.
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("header", ["", "Basic dXNlcjpwYXNz", "Bearer", "not-a-scheme token"])
def test_malformed_authorization_headers_are_401(client: TestClient, header: str) -> None:
    response = client.get("/protected", headers={"Authorization": header})
    assert response.status_code == 401


def test_a_token_the_library_rejects_is_401(client: TestClient) -> None:
    def explode(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("Signature mismatch for kid=abc123")

    with patch("google.oauth2.id_token.verify_oauth2_token", explode):
        response = _get(client, "forged")

    assert response.status_code == 401
    # The library's reason must not travel: it tells a forger how they were
    # caught. Asserting on the kid rather than a word like "signature" keeps
    # this from colliding with the generic message's own wording.
    assert "abc123" not in response.json()["detail"]


def test_a_wrong_issuer_is_401_and_not_a_500(client: TestClient) -> None:
    """
    GoogleAuthError is not a ValueError.

    verify_oauth2_token raises it for a bad issuer specifically. Catching only
    ValueError would let it escape the dependency as an unhandled 500, which
    reads to the caller as a broken server rather than a refused token.
    """
    from google.auth import exceptions as google_exceptions

    def wrong_issuer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise google_exceptions.GoogleAuthError("Wrong issuer.")

    with patch("google.oauth2.id_token.verify_oauth2_token", wrong_issuer):
        response = _get(client, "wrong-issuer")

    assert response.status_code == 401


def test_unverified_email_is_refused_even_when_allowlisted(client: TestClient) -> None:
    """
    The library does not check email_verified, so this module must.

    Without it, an account that merely claims an address satisfies an allowlist
    built out of addresses.
    """
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email_verified=False))):
        response = _get(client, "unverified")

    assert response.status_code == 401


def test_a_token_without_an_email_claim_is_refused(client: TestClient) -> None:
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email=""))):
        assert _get(client, "no-email").status_code == 401


# ── Authorisation failure is 403, not 401 ───────────────
def test_a_verified_stranger_is_403(client: TestClient) -> None:
    """
    Signing in worked; this deployment just does not serve them.

    A 401 here would send the dashboard around a re-login loop against a wall
    it can never get past, which is why the two codes are kept distinct.
    """
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email="stranger@example.com"))):
        response = _get(client, "stranger")

    assert response.status_code == 403
    assert "stranger@example.com" in response.json()["detail"]


# ── The cache ───────────────────────────────────────────
def test_a_verified_token_is_not_re_verified(client: TestClient) -> None:
    """Otherwise every request makes an outbound call to Google before doing any work."""
    calls = 0

    def counting(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _claims()

    with patch("google.oauth2.id_token.verify_oauth2_token", counting):
        for _ in range(3):
            assert _get(client, "same-token").status_code == 200

    assert calls == 1


def test_the_cache_never_outlives_the_token(client: TestClient) -> None:
    """
    The entry expires with the token, not five minutes after it.

    A TTL applied blindly would keep an already-expired token working.
    """
    now = time.time()

    # Expires in one second — far inside the five-minute TTL, so if the TTL
    # were applied blindly this entry would survive the check below.
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(exp=now + 1))):
        auth.verify_google_id_token("briefly-valid")

    assert auth._cached(auth._cache_key("briefly-valid")) is not None

    with patch.object(time, "time", lambda: now + 5):
        assert auth._cached(auth._cache_key("briefly-valid")) is None


def test_distinct_tokens_do_not_share_a_cache_entry(client: TestClient) -> None:
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims())):
        assert _get(client, "token-a").status_code == 200

    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email="stranger@example.com"))):
        assert _get(client, "token-b").status_code == 403


# ── Public routes ───────────────────────────────────────
def test_auth_config_is_reachable_without_a_token(client: TestClient) -> None:
    response = _get(client, path="/auth/config")
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "client_id": CLIENT_ID}


def test_auth_config_reveals_no_client_id_when_disabled(client: TestClient) -> None:
    with patch.object(core_config, "AUTH_ENABLED", False):
        assert client.get("/auth/config").json() == {"enabled": False, "client_id": ""}


def test_health_is_reachable_without_a_token() -> None:
    """
    /health must answer 200 with auth ON and no credentials.

    docker-compose.yml probes it to decide service_healthy, and the web
    container's depends_on waits on that. Guarding it makes the api container
    permanently unhealthy and the dashboard never starts — a failure that looks
    like a broken frontend and would be hunted far from here.

    Builds the real app rather than the fixture's stub, because what is under
    test is create_app()'s router wiring: admin_router deliberately does not
    take a blanket dependency, and this is what proves it.
    """
    import src.api.main as main_module

    @asynccontextmanager
    async def fake_checkpointer() -> Any:
        yield object()

    with (
        patch.object(main_module, "async_checkpointer", fake_checkpointer),
        patch.object(main_module, "build_research_graph", lambda checkpointer=None: object()),
        patch.object(main_module, "_connect_qdrant", lambda: (None, "unavailable")),
        patch("src.vectorstore.collections.ensure_collections", lambda: {}),
    ):
        app = main_module.create_app()
        with TestClient(app) as test_client:
            assert test_client.get("/health").status_code == 200
            assert test_client.get("/auth/config").status_code == 200

            # ...while its neighbours in the same router are guarded.
            assert test_client.get("/admin/budgets").status_code == 401
            assert test_client.get("/admin/config").status_code == 401

            # ...and so is every other router.
            assert test_client.get("/watchlist").status_code == 401
            assert test_client.get("/research/runs").status_code == 401
            assert test_client.get("/monitor/alerts").status_code == 401
            assert test_client.post("/monitor/cycles", json={"warmup": False}).status_code == 401


def _build_real_app() -> Any:
    """The real create_app(), with only the external services stubbed out."""
    import src.api.main as main_module

    @asynccontextmanager
    async def fake_checkpointer() -> Any:
        yield object()

    return main_module, fake_checkpointer


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_the_api_schema_is_not_public_when_auth_is_on(path: str) -> None:
    """
    FastAPI registers these itself, outside every router.

    So the router-level dependencies do not reach them, and without explicitly
    switching them off a guarded deployment still hands its entire API surface
    — all 17 paths — to anyone who asks. Guarding them instead is not an
    option: /docs is reached by typing a URL and a browser navigation cannot
    carry an Authorization header.
    """
    main_module, fake_checkpointer = _build_real_app()

    with (
        patch.object(main_module, "async_checkpointer", fake_checkpointer),
        patch.object(main_module, "build_research_graph", lambda checkpointer=None: object()),
        patch.object(main_module, "_connect_qdrant", lambda: (None, "unavailable")),
        patch("src.vectorstore.collections.ensure_collections", lambda: {}),
    ):
        with TestClient(main_module.create_app()) as test_client:
            assert test_client.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
def test_the_api_schema_stays_available_when_auth_is_off(path: str) -> None:
    """Turning auth off is also how you get the interactive docs back locally."""
    main_module, fake_checkpointer = _build_real_app()

    with (
        patch.object(core_config, "AUTH_ENABLED", False),
        patch.object(main_module, "async_checkpointer", fake_checkpointer),
        patch.object(main_module, "build_research_graph", lambda checkpointer=None: object()),
        patch.object(main_module, "_connect_qdrant", lambda: (None, "unavailable")),
        patch("src.vectorstore.collections.ensure_collections", lambda: {}),
    ):
        with TestClient(main_module.create_app()) as test_client:
            assert test_client.get(path).status_code == 200


# ── Auth switched off ───────────────────────────────────
def test_disabled_auth_admits_everyone_as_the_sentinel(client: TestClient) -> None:
    """
    The default. The CLIs, the eval harness and the test suite all depend on it.

    The sentinel is a real address rather than an empty string so that "who ran
    this" in a log never reads as a missing field when the answer is "auth was
    off".
    """
    with patch.object(core_config, "AUTH_ENABLED", False):
        response = _get(client)

    assert response.status_code == 200
    assert response.json() == {"user": auth.ANONYMOUS_USER}


# ── Startup validation ──────────────────────────────────
def test_validation_passes_when_fully_configured() -> None:
    core_config.validate_auth_config()


def test_validation_rejects_a_missing_client_id() -> None:
    with patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", ""), pytest.raises(ConfigurationError):
        core_config.validate_auth_config()


def test_validation_rejects_an_empty_allowlist() -> None:
    """An empty allowlist refuses every account while the API reports itself healthy."""
    with patch.object(core_config, "AUTH_ALLOWED_EMAILS", frozenset()), pytest.raises(ConfigurationError):
        core_config.validate_auth_config()


def test_validation_is_silent_when_auth_is_off() -> None:
    with patch.object(core_config, "AUTH_ENABLED", False), patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", ""):
        core_config.validate_auth_config()
