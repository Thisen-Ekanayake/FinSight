# ═══════════════════════════════════════════════════════
# FinSight — Tests: Authentication
# ═══════════════════════════════════════════════════════
#
# Nothing here reaches Google. verify_oauth2_token is replaced, so these test
# OUR half — the tier split, the 401/403 codes, the email_verified rule, the
# quota key, the cache — and not the library's signature checking, which is
# not ours to re-test.
#
# ══ AUTH_ALLOWED_EMAILS IS NOT AN ADMISSION LIST ANY MORE ══
#   It used to be: a verified account not named in it got a 403 and nothing
#   else. It is now the UNLIMITED TIER. Any verified Google account is
#   admitted and metered instead (src/api/quota.py, tests/test_quota.py), and
#   only the routes that spend unbounded money or disclose configuration still
#   demand membership. Several tests below assert the new direction where they
#   once asserted the old one, and say so where they do.
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
    """
    A minimal app: one route per tier, echoing the caller, plus /auth/config.

    /protected echoes the whole Identity rather than just the email, because
    the tier and the subject are now load-bearing — the first decides what a
    caller may reach, the second is the quota key.
    """
    app = FastAPI()

    @app.get("/protected")
    def protected(identity: auth.Identity = Depends(auth.require_identity)) -> dict[str, Any]:
        return {"user": identity.email, "subject": identity.subject, "unlimited": identity.unlimited}

    @app.get("/reserved")
    def reserved(user: str = Depends(auth.require_unlimited_user)) -> dict[str, str]:
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
    assert response.json()["user"] == ALLOWED


def test_email_is_matched_case_insensitively(client: TestClient) -> None:
    """Google may return the address in any case; the unlimited tier is lowercased."""
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email="Owner@Example.COM"))):
        response = _get(client, "mixed-case")

    assert response.status_code == 200
    assert response.json()["user"] == ALLOWED
    # The case-folding must reach the tier match too, not just the display name.
    assert response.json()["unlimited"] is True


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


# ── A verified stranger is a customer, not an intruder ──
def test_a_verified_stranger_is_admitted_and_metered(client: TestClient) -> None:
    """
    The free tier's central premise: anyone with a Google account gets in.

    This asserted 403 before the free tier existed, when AUTH_ALLOWED_EMAILS
    was an admission list. It is now the unlimited tier, so a stranger is
    admitted and metered instead of refused. Whether they may spend anything
    is src/api/quota.py's question, not this module's.
    """
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email="stranger@example.com"))):
        response = _get(client, "stranger")

    assert response.status_code == 200
    assert response.json()["user"] == "stranger@example.com"
    assert response.json()["unlimited"] is False


def test_the_unlimited_tier_is_flagged(client: TestClient) -> None:
    """An allowlisted address is admitted the same way, but exempt from the meter."""
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims())):
        response = _get(client, "owner")

    assert response.status_code == 200
    assert response.json()["unlimited"] is True


def test_a_stranger_is_refused_the_unlimited_only_routes(client: TestClient) -> None:
    """
    Opening sign-in to everyone must not open the routes that spend or disclose.

    403, not 401: the sign-in worked and repeating it cannot help, which is
    why the dashboard deliberately does not bounce a 403 to the sign-in screen.
    """
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email="stranger@example.com"))):
        response = _get(client, "stranger", path="/reserved")

    assert response.status_code == 403
    assert "stranger@example.com" in response.json()["detail"]


# ── The quota key is the `sub` claim, never the email ───
def test_identity_is_keyed_by_the_sub_claim(client: TestClient) -> None:
    """
    A Google account can change its address; `sub` is stable.

    Keying the lifetime counter by email would hand out a fresh allowance for
    the price of an email change, which is the whole failure the key exists
    to prevent.
    """
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(sub="112233445566"))):
        response = _get(client, "with-sub")

    assert response.json()["subject"] == "112233445566"


def test_a_token_without_sub_falls_back_to_the_email(client: TestClient) -> None:
    """
    A real Google token always carries `sub`. If one somehow does not, the
    quota layer must still get a per-account key — an empty one would collapse
    every such caller onto a single shared counter.
    """
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims())):
        response = _get(client, "no-sub")

    assert response.json()["subject"] == f"email:{ALLOWED}"


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

    # Distinguished by WHO comes back, not by the status: both are 200 now
    # that a stranger is admitted, so a shared cache entry would be invisible
    # to a status-code assertion.
    with patch("google.oauth2.id_token.verify_oauth2_token", _returns(_claims(email="stranger@example.com"))):
        second = _get(client, "token-b")
    assert second.status_code == 200
    assert second.json()["user"] == "stranger@example.com"


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
    assert response.json()["user"] == auth.ANONYMOUS_USER
    # Unlimited, not merely admitted. If this ever flips, pytest itself starts
    # writing quota rows and the free tier meters the eval harness.
    assert response.json()["unlimited"] is True


# ── Startup validation ──────────────────────────────────
def test_validation_passes_when_fully_configured() -> None:
    core_config.validate_auth_config()


def test_validation_rejects_a_missing_client_id() -> None:
    with patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", ""), pytest.raises(ConfigurationError):
        core_config.validate_auth_config()


def test_an_empty_allowlist_warns_but_no_longer_fails(caplog: pytest.LogCaptureFixture) -> None:
    """
    This used to be fatal, when the allowlist decided admission.

    Now it only means nobody is exempt from the meter — a legitimate, if
    rarely intended, configuration. So it must warn loudly and still boot,
    because failing here would refuse to start a deployment that works.
    """
    with patch.object(core_config, "AUTH_ALLOWED_EMAILS", frozenset()), caplog.at_level("WARNING"):
        core_config.validate_auth_config()

    assert "AUTH_ALLOWED_EMAILS is empty" in caplog.text


def test_validation_rejects_a_negative_free_limit() -> None:
    """A negative allowance is a typo, and it is checked in both auth modes."""
    with patch.object(core_config, "FREE_QUERY_LIMIT", -1), pytest.raises(ConfigurationError):
        core_config.validate_auth_config()

    with (
        patch.object(core_config, "AUTH_ENABLED", False),
        patch.object(core_config, "FREE_QUERY_LIMIT", -1),
        pytest.raises(ConfigurationError),
    ):
        core_config.validate_auth_config()


def test_validation_is_silent_when_auth_is_off() -> None:
    with patch.object(core_config, "AUTH_ENABLED", False), patch.object(core_config, "GOOGLE_OAUTH_CLIENT_ID", ""):
        core_config.validate_auth_config()
