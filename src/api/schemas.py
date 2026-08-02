# ═══════════════════════════════════════════════════════
# FinSight — API Schemas
# ═══════════════════════════════════════════════════════
#
# Purpose : Pydantic request and response models — the API's contract, kept
#           separate from the TypedDicts the graph passes around internally.
#
# Public API:
#   QueryRequest, QueryResponse
#   ThreadResponse, ThreadStep, RunSummaryOut
#   HealthResponse, BudgetOut, ConfigOut
#
# ══ WHY NOT JUST RETURN THE TypedDicts ══
#   ResearchState is an internal shape that changes whenever the graph does —
#   Phase 6 and 7 both add keys. Serialising it directly would publish every
#   one of those changes as a breaking API change, and would leak internals
#   like repair_targets that no client should depend on. These models are the
#   translation layer, and the cost of maintaining them is exactly the point.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────────
class QueryRequest(BaseModel):
    """A research question."""

    query: str = Field(
        min_length=3,
        max_length=1000,
        description="Natural-language financial research question",
        examples=["How did Apple's gross margin trend over the last three fiscal years?"],
    )
    thread_id: str | None = Field(
        default=None,
        max_length=64,
        description="Reuse an existing checkpoint thread. Omit to start a new one.",
    )


# ── Response components ─────────────────────────────────
class CitationOut(BaseModel):
    """A pointer back to the primary source of a claim."""

    source_type: str
    source_id: str
    url: str
    as_of: str
    excerpt: str | None = None


class ConflictOut(BaseModel):
    """Two sources disagreeing about the same metric beyond tolerance."""

    metric: str
    ticker: str | None = None
    values: list[tuple[str, float]]
    chosen_source: str
    chosen_value: float
    rel_difference: float


class UnsupportedClaimOut(BaseModel):
    """A claim the verifier could not ground in tool output."""

    claim: str
    reason: str
    origin_agent: str
    ticker: str


class VerificationOut(BaseModel):
    """
    The verifier's report.

    ``citation_coverage`` is the headline metric — the share of numeric claims
    traceable to a value some tool actually returned.
    """

    citation_coverage: float
    passed: bool
    verified_count: int
    unsupported_claims: list[UnsupportedClaimOut]
    invalid_source_ids: list[str]


class ToolCallOut(BaseModel):
    """One external call, for the audit trail."""

    node: str
    tool: str
    provider_used: str
    cache_hit: bool
    latency_ms: int
    ok: bool


# ── Responses ───────────────────────────────────────────
class QueryResponse(BaseModel):
    """A completed research run."""

    thread_id: str
    query: str
    answer: str
    citations: list[CitationOut]
    conflicts: list[ConflictOut]
    verification: VerificationOut
    agents_used: list[str]
    tickers: list[str]
    branch_count: int
    repair_count: int
    errors: list[str]
    latency_ms: int


class ThreadStep(BaseModel):
    """
    One superstep from the checkpoint history.

    ``nodes`` is who executed to produce this state and ``next`` is who runs
    next. Repeated names are the point: ``["fundamentals", "fundamentals"]``
    is one Send per ticker in a single superstep, which is what makes a
    parallel fan-out legible rather than something you have to take on trust.
    """

    step: int
    nodes: list[str]
    next: list[str]
    findings_total: int
    citations_total: int
    created_at: str | None = None


class RunSummaryOut(BaseModel):
    """One row of the run history."""

    thread_id: str
    query: str
    citation_coverage: float
    verification_passed: bool
    repair_count: int
    agents_used: list[str]
    tickers: list[str]
    latency_ms: int
    created_at: str


class ThreadResponse(BaseModel):
    """
    A thread's full audit trail, replayed from the checkpointer.

    Not a log written alongside the run — the actual intermediate states the
    graph passed through, read back from disk.
    """

    thread_id: str
    query: str
    answer: str
    summary: RunSummaryOut | None = None
    verification: VerificationOut | None = None
    tool_calls: list[ToolCallOut]
    steps: list[ThreadStep]


# ── Admin ───────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Liveness plus the state of each dependency."""

    status: str
    environment: str
    llm_backend: str
    database: bool
    checkpointer: bool
    qdrant: bool
    qdrant_detail: str


class BudgetOut(BaseModel):
    """One provider's usage against its daily allowance."""

    provider: str
    day: str
    used: int
    limit: int
    remaining: int
    soft_limit_reached: bool
    exhausted: bool


class ConfigOut(BaseModel):
    """
    Effective configuration, secrets excluded.

    Every field here is a name, a URL, or a number. No key, token, or
    credential path is ever added to this model — an endpoint that reports
    configuration is exactly where one gets leaked by accident.
    """

    environment: str
    llm_backend: str
    model_flash: str
    model_pro: str
    rpm_limits: dict[str, int]
    qdrant_url: str
    embedding_model: str
    embedding_dim: int
    tracing_enabled: bool
    langsmith_project: str
    numeric_tolerance: float
    max_repair_attempts: int
    verify_qualitative_claims: bool
