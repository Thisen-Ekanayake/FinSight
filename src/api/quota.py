# ═══════════════════════════════════════════════════════
# FinSight — Free-tier metering
# ═══════════════════════════════════════════════════════
#
# Purpose : Decide whether a caller may run another research query, spend the
#           unit if so, and refuse with a payable error if not.
#
# Public API:
#   is_metered(identity)   -> bool
#   charge_query(identity) -> None      raises 402 when exhausted
#   refund_query(identity) -> None
#   read_quota(identity)   -> QuotaOut
#
# ══ WHY THIS SITS BETWEEN auth.py AND repository.py ══
#   auth.py answers "who is calling" and must not grow SQL. repository.py
#   answers "what does the database say" and must not grow HTTP status codes.
#   The translation between them — an exhausted counter becomes a 402 carrying
#   a contact URL — is this module, and it is the only place that knows both.
#
# ══ 402, NOT 403 OR 429 ══
#   403 already means "not permitted on this deployment", and the dashboard
#   deliberately does NOT bounce a 403 to the sign-in screen — it would loop.
#   Reusing it here would make "you have run out" indistinguishable from "you
#   may never do this", which is exactly the distinction the free tier sells.
#   429 would be a lie: nothing resets, so no Retry-After is honest. 402
#   Payment Required is unused elsewhere in this API and means what it says.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging

from fastapi import HTTPException, status

from src.api.auth import Identity
from src.api.schemas import QuotaExceededOut, QuotaOut
from src.core import config as core_config
from src.persistence.repository import consume_free_query, get_free_quota, refund_free_query

logger = logging.getLogger(__name__)


def is_metered(identity: Identity) -> bool:
    """
    Whether this caller spends units at all.

    Two independent conditions, deliberately. ``AUTH_ENABLED=false`` must be
    COMPLETELY unmetered — the CLIs, the eval harness and every test in the
    suite run headless with auth off, and a meter that engaged for them would
    write quota rows during ``pytest``. That is already encoded in the
    Identity require_identity builds, and it is asserted again here so the
    bypass does not rest on a single call site being right.
    """
    return core_config.AUTH_ENABLED and not identity.unlimited


def charge_query(identity: Identity) -> None:
    """
    Spend one free query, or refuse with 402.

    Raises
    ------
    HTTPException
        402 when the account has no allowance left, carrying the counts and
        the contact URL so the browser can render a call to action.
    """
    if not is_metered(identity):
        return

    limit = core_config.FREE_QUERY_LIMIT
    spent = consume_free_query(identity.subject, limit=limit, email=identity.email)
    if spent is not None:
        logger.info("Free query %d/%d for %s", spent["used"], spent["limit"], identity.email)
        return

    standing = get_free_quota(identity.subject, limit=limit)
    logger.info("Refused %s — free allowance spent (%d/%d)", identity.email, standing["used"], limit)
    raise HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail=QuotaExceededOut(
            message=(
                f"You have used all {limit} free queries on this account."
                if limit
                else "This deployment does not offer free queries."
            ),
            used=standing["used"],
            limit=limit,
            contact_url=core_config.CONTACT_URL,
        ).model_dump(),
    )


def refund_query(identity: Identity) -> None:
    """
    Give a unit back when the run died before producing an answer.

    A failed graph run really did spend LLM tokens, so billing for it would be
    defensible — but on a five-query trial, charging someone for our own crash
    is the wrong trade. A client that aborts mid-stream is NOT refunded: that
    path raises CancelledError rather than Exception, and refunding it would
    make "type, abort, retype" free forever.
    """
    if not is_metered(identity):
        return
    refund_free_query(identity.subject)
    logger.info("Refunded one free query to %s after a failed run", identity.email)


def read_quota(identity: Identity) -> QuotaOut:
    """Report this account's standing for GET /auth/quota, spending nothing."""
    limit = core_config.FREE_QUERY_LIMIT

    if not is_metered(identity):
        return QuotaOut(
            metered=False,
            used=0,
            limit=limit,
            remaining=limit,
            contact_url=core_config.CONTACT_URL,
        )

    standing = get_free_quota(identity.subject, limit=limit)
    return QuotaOut(
        metered=True,
        used=standing["used"],
        limit=standing["limit"],
        remaining=standing["remaining"],
        contact_url=core_config.CONTACT_URL,
    )
