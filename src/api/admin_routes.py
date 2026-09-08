# ═══════════════════════════════════════════════════════
# FinSight — Admin Routes
# ═══════════════════════════════════════════════════════
#
# Purpose : Operational visibility — is it up, what is it configured as, and
#           how much of each free tier is left today.
#
# Routes:
#   GET /health          liveness and per-dependency status        (PUBLIC)
#   GET /admin/budgets   today's API usage                (UNLIMITED TIER ONLY)
#   GET /admin/config    effective configuration          (UNLIMITED TIER ONLY)
#
# ══ /health IS PUBLIC AND MUST STAY THAT WAY ══
#   docker-compose.yml probes it to decide service_healthy, and the web
#   container's depends_on waits on that. Putting it behind require_user makes
#   the api container permanently unhealthy and the frontend never starts.
#   This is why the two /admin routes carry their own Depends rather than the
#   router taking a blanket one in create_app().
#
# ══ WHY THE ADMIN ROUTES NEED require_unlimited_user ══
#   require_user now admits any verified Google account, because the free tier
#   sells queries to strangers. These two routes report the deployment's own
#   spend and configuration, which is the operator's business and nobody
#   else's — so they ask for the unlimited tier by name.
#
# ══ /config NEVER RETURNS A SECRET ══
#   An endpoint whose job is to report configuration is the single easiest
#   place to leak a key: someone adds a field for debugging and it ships. So
#   the response model lists every field explicitly and holds only names,
#   URLs, and numbers. It is built by naming fields, never by iterating over
#   the config module — an iteration would pick up GOOGLE_API_KEY the moment
#   somebody added one. NOTIFICATION_SINKS is a list of sink NAMES
#   ("console", "file", "email"); the SMTP credentials behind the email sink
#   live in src/monitor/config.py and are not reachable from here.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text

from src.api.auth import require_unlimited_user
from src.api.schemas import BudgetOut, ConfigOut, HealthResponse
from src.core import config as core_config
from src.core.tracing import is_tracing_enabled
from src.monitor.config import MONITOR_CADENCE_HOURS, NOTIFICATION_SINKS
from src.persistence.db import get_engine
from src.persistence.repository import get_budget_status
from src.research.config import MAX_REPAIR_ATTEMPTS, NUMERIC_REL_TOLERANCE, VERIFY_QUALITATIVE_CLAIMS
from src.vectorstore.config import COLLECTION_ALERTS, COLLECTION_FILINGS, TAU_HIGH, TAU_LOW

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


def _collection_status(client: object | None) -> tuple[dict[str, bool] | None, str | None]:
    """
    Ask whether each expected collection exists, live.

    Reachability is not readiness. A Qdrant that is up but holds no collections
    answers every connection check happily while the monitor dies on a 404 from
    its dedup lookup — which is exactly how a real outage stayed invisible here:
    ``qdrant_detail`` said "connected" against an instance with zero
    collections. So this asks about the collections rather than the socket.

    It also re-checks on every call, where ``qdrant_detail`` alone is a string
    frozen at startup — a Qdrant that died an hour ago still reads "connected".

    Returns
    -------
    tuple
        ``(presence_or_None, detail_suffix_or_None)``. None presence means
        Qdrant could not be asked at all, which is distinct from a definite
        answer that something is missing.
    """
    if client is None:
        return None, None

    try:
        present = {
            name: bool(client.collection_exists(name))  # type: ignore[attr-defined]
            for name in (COLLECTION_FILINGS, COLLECTION_ALERTS)
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant collection check failed: %s", exc)
        return None, f"collection check failed: {type(exc).__name__}"

    missing = sorted(name for name, exists in present.items() if not exists)
    if missing:
        return present, f"MISSING collections: {', '.join(missing)}"
    return present, None


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependency status")
async def health(request: Request) -> HealthResponse:
    """
    Report liveness and the state of each dependency.

    Returns 200 even when Qdrant is down: the API still answers every numeric
    question without it, and only the filings specialist is degraded. A health
    check that fails the whole service for a partial outage teaches its
    operator to ignore it. Missing collections are reported the same way and
    for the same reason — loudly in ``qdrant_collections`` and
    ``qdrant_detail``, without flipping ``status``.
    """
    from src.monitor.config import MONITOR_SCHEDULER_ENABLED
    from src.monitor.scheduler import next_run_at

    qdrant_client = getattr(request.app.state, "qdrant", None)
    checkpointer = getattr(request.app.state, "checkpointer", None)
    scheduler = getattr(request.app.state, "monitor_scheduler", None)
    database_ok = _database_ok()

    collections, collection_detail = _collection_status(qdrant_client)
    qdrant_detail = getattr(request.app.state, "qdrant_detail", "unknown")
    if collection_detail:
        qdrant_detail = f"{qdrant_detail}, {collection_detail}"

    return HealthResponse(
        status="ok" if database_ok else "degraded",
        environment=core_config.ENVIRONMENT,
        llm_backend=core_config.GEMINI_BACKEND,
        database=database_ok,
        checkpointer=checkpointer is not None,
        qdrant=qdrant_client is not None,
        qdrant_detail=qdrant_detail,
        qdrant_collections=collections,
        scheduler_enabled=MONITOR_SCHEDULER_ENABLED,
        scheduler_next_run_at=next_run_at(scheduler),
    )


@router.get(
    "/admin/budgets",
    response_model=list[BudgetOut],
    summary="Today's API usage",
    dependencies=[Depends(require_unlimited_user)],
)
async def budgets() -> list[BudgetOut]:
    """
    Report each provider's usage against its daily allowance.

    Every budgeted provider appears even at zero calls — an empty list would
    read as "budgets are not being tracked", which is a different and much
    worse thing than "nothing has been spent".
    """
    return [BudgetOut(**status) for status in get_budget_status()]


@router.get(
    "/admin/config",
    response_model=ConfigOut,
    summary="Effective configuration",
    dependencies=[Depends(require_unlimited_user)],
)
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
        dedup_tau_high=TAU_HIGH,
        dedup_tau_low=TAU_LOW,
        monitor_cadence_hours=MONITOR_CADENCE_HOURS,
        notification_sinks=list(NOTIFICATION_SINKS),
    )
