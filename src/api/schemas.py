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


class SeriesPointOut(BaseModel):
    """
    One numeric finding, shaped for plotting.

    ``provider`` travels with every point on purpose. The fundamentals chain
    falls through EDGAR -> yfinance -> FMP, so two points on one line can have
    different provenance, and a chart that hid that would imply a uniformity
    the data does not have. The UI labels the source rather than drawing a
    line as if every point were filed.
    """

    ticker: str
    metric: str
    period: str
    value: float
    unit: str | None = None
    provider: str | None = None
    confidence: float = 1.0


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
    # Chartable subset of the run's findings. Empty for questions that produce
    # no numeric time series, which is most non-fundamentals questions.
    series: list[SeriesPointOut] = []


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


# ── Auth ────────────────────────────────────────────────
class AuthConfigOut(BaseModel):
    """
    What the browser needs in order to sign in.

    Holds no secret. ``client_id`` is public by construction — it appears in
    every OAuth flow — and is empty when ``enabled`` is false so that a
    deployment running without auth reveals nothing about one that does.
    """

    enabled: bool
    client_id: str


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
    # {collection_name: exists}. None when Qdrant itself could not be asked,
    # which is a different thing from "asked, and the collections are missing".
    qdrant_collections: dict[str, bool] | None = None
    scheduler_enabled: bool
    scheduler_next_run_at: str | None = None


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

    # ══ WHY THE MONITORING NUMBERS ARE PUBLISHED ══
    #   The dashboard explains its own dedup decisions in prose — "folded
    #   above 0.89, between 0.78 and 0.89 it is merged as the same event
    #   rather than dropped". Hardcoding those two numbers in the frontend
    #   would make the explanation silently wrong the first time anyone
    #   re-tuned them, and they must be re-tuned from scratch whenever
    #   EMBEDDING_MODEL changes (src/vectorstore/config.py). Reading them
    #   from here is the only way that sentence stays true.
    dedup_tau_high: float
    dedup_tau_low: float
    monitor_cadence_hours: int
    notification_sinks: list[str]


# ── Monitoring (Phase 6) ────────────────────────────────
class WatchItemOut(BaseModel):
    """One watched ticker."""

    ticker: str
    company_name: str
    warmed_up: bool
    added_at: str
    last_checked: dict[str, str] = Field(
        default_factory=dict,
        description="Per-monitor watermark, e.g. {'filing': '2026-08-03T09:31:00+00:00'}",
    )


class WatchItemRequest(BaseModel):
    """Add a ticker to the watchlist."""

    ticker: str = Field(min_length=1, max_length=16, description="US-listed symbol")
    company_name: str = Field(default="", max_length=128, description="Resolved from EDGAR when omitted")


class AlertOut(BaseModel):
    """
    One fired alert.

    ``occurrence_count`` is how many times the event has been SEEN, not how
    many times it was reported — the whole point of the dedup engine is that
    those two numbers differ.
    """

    alert_id: str
    cycle_id: str
    ticker: str
    alert_type: str
    severity: str
    status: str
    headline: str
    detail: str
    canonical_text: str
    occurrence_count: int
    first_seen_at: str
    last_seen_at: str
    fired_at: str
    parent_alert_id: str | None = None
    evidence: list[CitationOut] = Field(default_factory=list)


class SuppressionOut(BaseModel):
    """
    A candidate that was NOT reported, with the alert it collapsed into.

    Exposed rather than hidden. A dedup engine whose decisions are invisible
    is indistinguishable from one that is dropping alerts through a bug, and
    this is the endpoint that tells the two apart.
    """

    ticker: str
    alert_type: str
    headline: str
    parent_alert_id: str
    parent_headline: str
    score: float
    reason: str


class DedupDecisionOut(BaseModel):
    """One dedup decision with the score behind it — the Phase 7 sweep's input."""

    cycle_id: str
    ticker: str
    alert_type: str
    severity: str
    decision: str
    reason: str
    dedup_key: str
    candidate_text: str
    parent_alert_id: str | None = None
    parent_text: str
    score: float
    decided_at: str


class CycleOut(BaseModel):
    """One monitoring cycle, summarised."""

    cycle_id: str
    status: str
    warmup: bool
    tickers: list[str]
    candidate_count: int
    fired_count: int
    suppressed_count: int
    merged_count: int
    error_count: int
    api_call_count: int
    started_at: str
    duration_ms: int


class CycleRunRequest(BaseModel):
    """Trigger one cycle on demand."""

    warmup: bool = Field(default=False, description="Observe-only: index candidates, report nothing")
    tickers: list[str] = Field(default_factory=list, description="Override the stored watchlist for this run")


class CycleRunResponse(BaseModel):
    """
    The outcome of one cycle, including what it chose not to tell you.

    ``status`` is "COMPLETE" for an ordinary cycle. "PENDING_APPROVAL" means
    the graph paused before dispatcher_node ran — ``fired``/``merged`` here
    are the PRE-decision state (a HIGH alert still carries status
    PENDING_APPROVAL) and ``pending_approval`` is what a human has to resolve
    via ``POST /monitor/cycles/{cycle_id}/resume`` before anything is
    actually delivered.
    """

    cycle_id: str
    status: str
    warmup: bool
    candidate_count: int
    fired: list[AlertOut]
    merged: list[AlertOut]
    suppressed: list[SuppressionOut]
    pending_approval: list[AlertOut] = Field(default_factory=list)
    errors: list[str]
    duration_ms: int


class ResumeCycleRequest(BaseModel):
    """
    Decide every pending alert on a paused cycle.

    ``decisions`` must cover EXACTLY the alert ids the cycle is waiting on —
    see resume_cycle's docstring in src/monitor/graph.py for why a partial or
    mismatched dict is rejected rather than partially applied.
    """

    decisions: dict[str, str] = Field(description='{alert_id: "approve" | "reject"}, one entry per pending alert')
