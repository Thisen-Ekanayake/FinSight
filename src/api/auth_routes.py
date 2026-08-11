# ═══════════════════════════════════════════════════════
# FinSight — Auth Routes
# ═══════════════════════════════════════════════════════
#
# Purpose : Tell the browser how to sign in, before anyone has.
#
# Routes:
#   GET /auth/config   whether auth is on, and the client ID to use
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
# ══ THIS ROUTE IS PUBLIC, NECESSARILY ══
#   It is what an unauthenticated browser reads in order to become an
#   authenticated one. It carries no secret and no state.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import AuthConfigOut
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
