# ═══════════════════════════════════════════════════════
# FinSight — Core Package
# ═══════════════════════════════════════════════════════
#
# Cross-cutting infrastructure: configuration, LLM access, tracing, shared
# schemas, logging, and the exception hierarchy.
#
# Public API:
#   configure_logging, configure_tracing, trace_metadata
#   get_llm, ModelTier
#   Citation, AgentFinding, ToolCallRecord, Conflict, make_citation
#   FinSightError and subclasses
#
# Note: get_llm is exposed as a lazy wrapper so that importing this package
# does not pull in langchain_google_genai (and its transitive gRPC stack) for
# callers that only need config or schemas.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.errors import (
    BudgetExhausted,
    ConfigurationError,
    DataSourceError,
    FinSightError,
    InfrastructureError,
    MissingCredentialError,
    QdrantIsolationError,
    RateLimitExceeded,
    VerificationFailed,
)
from src.core.logging_setup import configure_logging
from src.core.schemas import (
    AgentFinding,
    Citation,
    Conflict,
    Severity,
    SourceType,
    ToolCallRecord,
    make_citation,
)
from src.core.tracing import configure_tracing, is_tracing_enabled, trace_metadata

if TYPE_CHECKING:  # pragma: no cover
    from src.core.config import ModelTier


def get_llm(tier: str = "flash", **kwargs: Any) -> Any:
    """Lazy re-export of src.core.llm.get_llm — see that module for docs."""
    from src.core.llm import get_llm as _get_llm

    return _get_llm(tier, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "configure_logging",
    "configure_tracing",
    "is_tracing_enabled",
    "trace_metadata",
    "get_llm",
    "ModelTier",
    "Citation",
    "AgentFinding",
    "ToolCallRecord",
    "Conflict",
    "SourceType",
    "Severity",
    "make_citation",
    "FinSightError",
    "ConfigurationError",
    "MissingCredentialError",
    "InfrastructureError",
    "QdrantIsolationError",
    "DataSourceError",
    "RateLimitExceeded",
    "BudgetExhausted",
    "VerificationFailed",
]
