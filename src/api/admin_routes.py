# ═══════════════════════════════════════════════════════
# FinSight — Admin Routes
# ═══════════════════════════════════════════════════════
#
# Purpose : Operational visibility — is it up, what is it configured as, and
#           how much of each free tier is left today.
#
# Routes:
#   GET /health          liveness and per-dependency status
#   GET /admin/budgets   today's API usage against each daily allowance
#   GET /admin/config    effective configuration, secrets excluded
#
# ══ /config NEVER RETURNS A SECRET ══
#   An endpoint whose job is to report configuration is the single easiest
#   place to leak a key: someone adds a field for debugging and it ships. So
#   the response model lists every field explicitly and holds only names,
#   URLs, and numbers. It is built by naming fields, never by iterating over
#   the config module — an iteration would pick up GOOGLE_API_KEY the moment
#   somebody added one.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from sqlalchemy import text

from src.api.schemas import BudgetOut, ConfigOut, HealthResponse
from src.core import config as core_config
from src.core.tracing import is_tracing_enabled
from src.persistence.db import get_engine
from src.persistence.repository import get_budget_status
from src.research.config import MAX_REPAIR_ATTEMPTS, NUMERIC_REL_TOLERANCE, VERIFY_QUALITATIVE_CLAIMS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])


def _database_ok() -> bool:
    """Round-trip a trivial query rather than trusting that an engine exists."""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database health check failed: %s", exc)
        return False


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependency status")
async def health(request: Request) -> HealthResponse:
    """
    Report liveness and the state of each dependency.

    Returns 200 even when Qdrant is down: the API still answers every numeric
    question without it, and only the filings specialist is degraded. A health
    check that fails the whole service for a partial outage teaches its
    operator to ignore it.
    """
    qdrant_client = getattr(request.app.state, "qdrant", None)
    checkpointer = getattr(request.app.state, "checkpointer", None)
    database_ok = _database_ok()

    return HealthResponse(
        status="ok" if database_ok else "degraded",
        environment=core_config.ENVIRONMENT,
        llm_backend=core_config.GEMINI_BACKEND,
        database=database_ok,
        checkpointer=checkpointer is not None,
        qdrant=qdrant_client is not None,
        qdrant_detail=getattr(request.app.state, "qdrant_detail", "unknown"),
    )


@router.get("/admin/budgets", response_model=list[BudgetOut], summary="Today's API usage")
async def budgets() -> list[BudgetOut]:
    """
    Report each provider's usage against its daily allowance.

    Every budgeted provider appears even at zero calls — an empty list would
    read as "budgets are not being tracked", which is a different and much
    worse thing than "nothing has been spent".
    """
    return [BudgetOut(**status) for status in get_budget_status()]


@router.get("/admin/config", response_model=ConfigOut, summary="Effective configuration")
async def effective_config() -> ConfigOut:
    """
    Report the configuration actually in force.

    Answers "is it really pointed at 6335?" and "did GEMINI_BACKEND take?"
    without reading a .env over someone's shoulder. Contains no credentials.
    """
    return ConfigOut(
        environment=core_config.ENVIRONMENT,
        llm_backend=core_config.GEMINI_BACKEND,
        model_flash=core_config.GEMINI_MODEL_FLASH,
        model_pro=core_config.GEMINI_MODEL_PRO,
        rpm_limits=dict(core_config.GEMINI_RPM),
        qdrant_url=core_config.QDRANT_URL,
        embedding_model=core_config.EMBEDDING_MODEL,
        embedding_dim=core_config.EMBEDDING_DIM,
        tracing_enabled=is_tracing_enabled(),
        langsmith_project=core_config.LANGSMITH_PROJECT,
        numeric_tolerance=NUMERIC_REL_TOLERANCE,
        max_repair_attempts=MAX_REPAIR_ATTEMPTS,
        verify_qualitative_claims=VERIFY_QUALITATIVE_CLAIMS,
    )
