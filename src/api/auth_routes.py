# ═══════════════════════════════════════════════════════
# FinSight — Auth Routes
# ═══════════════════════════════════════════════════════
#
# Purpose : Tell the browser how to sign in, and what its allowance is once it has.
#
# Routes:
#   GET /auth/config   whether auth is on, and the client ID to use  (public)
#   GET /auth/quota    this account's free-query standing            (authenticated)
#
# ══ WHY THE CLIENT ID IS SERVED, NOT BAKED INTO THE BUNDLE ══
#   Vite inlines env vars at BUILD time, which is exactly why
#   frontend/Dockerfile leaves VITE_API_URL unset: one image has to run on any
#   host. A VITE_GOOGLE_CLIENT_ID build arg would give that up and tie every
#   image to one Google project. Serving the ID at runtime keeps the bundle
#   host-agnostic and lets the same image run against a deployment with auth
#   off entirely.
#
#   A client ID is not a secret. It is in the URL of every OAuth flow ever
#   made — Google's security model rests on the authorised-origins list and
#   the token signature, never on the ID being hidden.
#
# ══ /config IS PUBLIC, NECESSARILY — /quota IS NOT ══
#   /config is what an unauthenticated browser reads in order to become an
#   authenticated one, so this router is included WITHOUT the app-wide guard
#   (see src/api/main.py). /quota therefore carries its own dependency in its
#   signature. Same shape as admin_routes.py, where an otherwise-guarded
#   router keeps /health open for the healthcheck.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from fastapi import APIRouter

from src.api.auth import CurrentIdentity
from src.api.quota import read_quota
from src.api.schemas import AuthConfigOut, QuotaOut
from src.core import config as core_config

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/config", response_model=AuthConfigOut, summary="How to sign in")
async def auth_config() -> AuthConfigOut:
    """
    Report whether this deployment requires a sign-in, and against which client.

    ``enabled: false`` is a complete answer — the dashboard then skips Google
    entirely rather than rendering a button that cannot work, which is what
    makes `npm run dev` against a local API need no Google setup at all.
    """
    return AuthConfigOut(
        enabled=core_config.AUTH_ENABLED,
        client_id=core_config.GOOGLE_OAUTH_CLIENT_ID if core_config.AUTH_ENABLED else "",
    )


@router.get("/quota", response_model=QuotaOut, summary="This account's free-query allowance")
async def auth_quota(identity: CurrentIdentity) -> QuotaOut:
    """
    Report what this account has left, without spending any of it.

    Exists so the dashboard can say "3 of 5 free queries left" BEFORE the
    user commits to a question. Discovering the limit only by hitting a 402
    would mean the last free query and the first refused one look identical
    until it is too late to choose differently.
    """
    return read_quota(identity)
