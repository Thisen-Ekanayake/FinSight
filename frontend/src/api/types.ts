// ═══════════════════════════════════════════════════════
// FinSight web — API contract types
// ═══════════════════════════════════════════════════════
//
// Purpose : A one-to-one TypeScript mirror of src/api/schemas.py. Field names
//           and optionality match the Pydantic models exactly; nothing is
//           renamed on the way in.
//
// ══ WHY MIRROR RATHER THAN GENERATE ══
//   FastAPI publishes an OpenAPI document, so these could be generated. They
//   are hand-written because the generated output for `list[tuple[str, float]]`
//   and for the several `X | None` fields is noisier than the thing it
//   replaces, and because a hand-written mirror fails LOUDLY at review time
//   when schemas.py changes — a regenerated file just quietly changes shape.
//   The Python docstrings that explain a field's meaning are carried across
//   where the meaning is not obvious from the name.
// ═══════════════════════════════════════════════════════

export type Severity = 'HIGH' | 'MED' | 'LOW';

/** A pointer back to the primary source of a claim. */
export interface Citation {
  source_type: string;
  source_id: string;
  url: string;
  as_of: string;
  excerpt: string | null;
}

/** Two sources disagreeing about the same metric beyond tolerance. */
export interface Conflict {
  metric: string;
  ticker: string | null;
  values: [string, number][];
  chosen_source: string;
  chosen_value: number;
  rel_difference: number;
}

/** A claim the verifier could not ground in tool output. */
export interface UnsupportedClaim {
  claim: string;
  reason: string;
  origin_agent: string;
  ticker: string;
}

/**
 * The verifier's report. `citation_coverage` is the headline metric — the
 * share of numeric claims traceable to a value some tool actually returned.
 */
export interface Verification {
  citation_coverage: number;
  passed: boolean;
  verified_count: number;
  unsupported_claims: UnsupportedClaim[];
  invalid_source_ids: string[];
}

/** One external call, for the audit trail. */
export interface ToolCall {
  node: string;
  tool: string;
  provider_used: string;
  cache_hit: boolean;
  latency_ms: number;
  ok: boolean;
}

/**
 * One numeric finding, shaped for plotting.
 *
 * `provider` travels with every point because the fundamentals chain falls
 * through EDGAR -> yfinance -> FMP: two points on the same line can have
 * different provenance, and a chart that hid that would imply a uniformity the
 * data does not have.
 */
export interface SeriesPoint {
  ticker: string;
  metric: string;
  period: string;
  value: number;
  unit: string | null;
  provider: string | null;
  confidence: number;
}

/** A completed research run. */
export interface QueryResponse {
  thread_id: string;
  query: string;
  answer: string;
  citations: Citation[];
  conflicts: Conflict[];
  verification: Verification;
  agents_used: string[];
  tickers: string[];
  branch_count: number;
  repair_count: number;
  errors: string[];
  latency_ms: number;
  /** Empty for questions that produce no numeric time series — most non-fundamentals ones. */
  series: SeriesPoint[];
}

/**
 * One superstep from the checkpoint history. Repeated names in `nodes` are
 * the point: ["fundamentals", "fundamentals"] is one Send per ticker in a
 * single superstep — the evidence that the fan-out was parallel.
 */
export interface ThreadStep {
  step: number;
  nodes: string[];
  next: string[];
  findings_total: number;
  citations_total: number;
  created_at: string | null;
}

/** One row of the run history. */
export interface RunSummary {
  thread_id: string;
  query: string;
  citation_coverage: number;
  verification_passed: boolean;
  repair_count: number;
  agents_used: string[];
  tickers: string[];
  latency_ms: number;
  created_at: string;
}

/** A thread's full audit trail, replayed from the checkpointer. */
export interface ThreadResponse {
  thread_id: string;
  query: string;
  answer: string;
  summary: RunSummary | null;
  verification: Verification | null;
  tool_calls: ToolCall[];
  steps: ThreadStep[];
}

/** Liveness plus the state of each dependency. */
export interface Health {
  status: string;
  environment: string;
  llm_backend: string;
  database: boolean;
  checkpointer: boolean;
  qdrant: boolean;
  qdrant_detail: string;
  scheduler_enabled: boolean;
  scheduler_next_run_at: string | null;
}

/** One provider's usage against its daily allowance. */
export interface Budget {
  provider: string;
  day: string;
  used: number;
  limit: number;
  remaining: number;
  soft_limit_reached: boolean;
  exhausted: boolean;
}

/** Effective configuration, secrets excluded. */
export interface Config {
  environment: string;
  llm_backend: string;
  model_flash: string;
  model_pro: string;
  rpm_limits: Record<string, number>;
  qdrant_url: string;
  embedding_model: string;
  embedding_dim: number;
  tracing_enabled: boolean;
  langsmith_project: string;
  numeric_tolerance: number;
  max_repair_attempts: number;
  verify_qualitative_claims: boolean;
  /** Similarity at or above which a candidate is folded away as a duplicate. */
  dedup_tau_high: number;
  /** Between tau_low and tau_high a candidate is merged, not dropped. */
  dedup_tau_low: number;
  monitor_cadence_hours: number;
  notification_sinks: string[];
}

/** One watched ticker. `last_checked` is a per-monitor watermark. */
export interface WatchItem {
  ticker: string;
  company_name: string;
  warmed_up: boolean;
  added_at: string;
  last_checked: Record<string, string>;
}

/**
 * One fired alert. `occurrence_count` is how many times the event has been
 * SEEN, not how many times it was reported — the whole point of the dedup
 * engine is that those two numbers differ.
 */
export interface Alert {
  alert_id: string;
  cycle_id: string;
  ticker: string;
  alert_type: string;
  severity: Severity;
  status: string;
  headline: string;
  detail: string;
  canonical_text: string;
  occurrence_count: number;
  first_seen_at: string;
  last_seen_at: string;
  fired_at: string;
  parent_alert_id: string | null;
  evidence: Citation[];
}

/** A candidate that was NOT reported, with the alert it collapsed into. */
export interface Suppression {
  ticker: string;
  alert_type: string;
  headline: string;
  parent_alert_id: string;
  parent_headline: string;
  score: number;
  reason: string;
}

/** One dedup decision with the score behind it. */
export interface DedupDecision {
  cycle_id: string;
  ticker: string;
  alert_type: string;
  severity: Severity;
  decision: string;
  reason: string;
  dedup_key: string;
  candidate_text: string;
  parent_alert_id: string | null;
  parent_text: string;
  score: number;
  decided_at: string;
}

/** One monitoring cycle, summarised. */
export interface Cycle {
  cycle_id: string;
  status: 'COMPLETE' | 'PENDING_APPROVAL' | string;
  warmup: boolean;
  tickers: string[];
  candidate_count: number;
  fired_count: number;
  suppressed_count: number;
  merged_count: number;
  error_count: number;
  api_call_count: number;
  started_at: string;
  duration_ms: number;
}

/**
 * The outcome of one cycle, including what it chose not to tell you.
 *
 * `status` is "COMPLETE" for an ordinary cycle. "PENDING_APPROVAL" means the
 * graph paused before dispatch — `fired`/`merged` are then the PRE-decision
 * state and `pending_approval` is what a human has to resolve.
 */
export interface CycleRunResponse {
  cycle_id: string;
  status: string;
  warmup: boolean;
  candidate_count: number;
  fired: Alert[];
  merged: Alert[];
  suppressed: Suppression[];
  pending_approval: Alert[];
  errors: string[];
  duration_ms: number;
}

/** `{alert_id: "approve" | "reject"}`, one entry per pending alert. */
export type Decisions = Record<string, 'approve' | 'reject'>;

// ── Streaming (POST /research/query/stream) ─────────────
//
// Frames are `start`, one `node` per completed node, then `final` — or
// `error`. See _stream_events in src/api/research_routes.py.

export interface StreamStart {
  thread_id: string;
  query: string;
}

export interface StreamNode {
  node: string;
  findings: number;
  errors: string[];
  plan: unknown;
  repairs: number;
}

export type StreamEvent =
  | { event: 'start'; data: StreamStart }
  | { event: 'node'; data: StreamNode }
  | { event: 'final'; data: QueryResponse }
  | { event: 'error'; data: { detail: string } };
