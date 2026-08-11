// ═══════════════════════════════════════════════════════
// FinSight web — HTTP client
// ═══════════════════════════════════════════════════════
//
// Purpose : Every call the dashboard makes to the FastAPI backend, in one
//           place — the TypeScript counterpart of src/ui/client.py, endpoint
//           for endpoint.
//
// Public API:
//   ApiError                            thrown on any non-2xx response
//   ask / askStream                     POST /research/query[/stream]
//   getThread / listRuns                GET  /research/threads/{id}, /research/runs
//   runCycle / resumeCycle              POST /monitor/cycles[/{id}/resume]
//   listCycles / getCycle / getPending  GET  /monitor/cycles...
//   listAlerts / listDecisions          GET  /monitor/alerts, /monitor/decisions
//   getWatchlist / addTicker / removeTicker
//   health / budgets / config
//
// ══ WHERE THE BASE URL COMES FROM ══
//   VITE_API_URL if the build sets one, otherwise the same-origin prefix
//   "/api" — which the Vite dev server proxies to :8000 and nginx proxies to
//   the api container. One code path for dev and prod, and no CORS preflight
//   in either. The backend's wide-open CORS still makes a direct
//   VITE_API_URL=http://host:8000 build work for anyone who wants it.
// ═══════════════════════════════════════════════════════

import type {
  Alert,
  AuthConfig,
  Budget,
  Config,
  Cycle,
  CycleRunResponse,
  DedupDecision,
  Decisions,
  Health,
  QueryResponse,
  RunSummary,
  StreamEvent,
  ThreadResponse,
  WatchItem,
} from './types';

export const API_BASE: string = (import.meta.env.VITE_API_URL ?? '/api').replace(/\/$/, '');

/** A non-2xx response, carrying the server's own `detail` message. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// ══ AUTH: ONE TOKEN, TWO FETCH SITES ══
//   The token lives here rather than in React state because askStream and
//   request are plain functions that views call directly — threading it
//   through every signature would touch every caller for no benefit.
//
//   It is deliberately NOT persisted to localStorage. A Google ID token is a
//   bearer credential valid for about an hour; parking it where any script on
//   the origin can read it buys a slightly shorter sign-in and pays for it in
//   the worst currency available. Losing it on refresh is the correct
//   trade — useAuth re-acquires silently when Google still has a session.

let authToken: string | null = null;
let onUnauthorized: (() => void) | null = null;

/** Set (or clear, with null) the bearer token sent on every subsequent call. */
export function setAuthToken(token: string | null): void {
  authToken = token;
}

/**
 * Register the callback fired when the API rejects our token.
 *
 * The token is cleared before the callback runs, so a handler that re-renders
 * cannot race an in-flight retry into a second 401.
 */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler;
}

function authHeaders(): Record<string, string> {
  return authToken ? { Authorization: `Bearer ${authToken}` } : {};
}

/**
 * Handle a 401 once, centrally.
 *
 * 401 means the token is missing, expired or invalid — all recoverable by
 * signing in again. 403 deliberately does NOT come through here: it means the
 * account is verified but not on this deployment's allowlist, so bouncing to
 * the sign-in screen would loop against a wall it can never get past.
 */
function handleUnauthorized(status: number): void {
  if (status !== 401) return;
  authToken = null;
  onUnauthorized?.();
}

type Query = Record<string, string | number | boolean | undefined | null>;

function withQuery(path: string, params?: Query): string {
  if (!params) return `${API_BASE}${path}`;
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') search.set(key, String(value));
  }
  const qs = search.toString();
  return `${API_BASE}${path}${qs ? `?${qs}` : ''}`;
}

/**
 * Make one call and return its parsed JSON body.
 *
 * Throws ApiError on any non-2xx response, and on a transport failure too —
 * both are the caller's problem to display, never an unhandled rejection.
 */
async function request<T>(
  method: string,
  path: string,
  opts: { params?: Query; body?: unknown; signal?: AbortSignal } = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(withQuery(path, opts.params), {
      method,
      headers: {
        ...authHeaders(),
        ...(opts.body === undefined ? {} : { 'Content-Type': 'application/json' }),
      },
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      signal: opts.signal,
    });
  } catch (exc) {
    if (exc instanceof DOMException && exc.name === 'AbortError') throw exc;
    throw new ApiError(`Could not reach the API at ${API_BASE} — is it running? (${String(exc)})`);
  }

  if (!response.ok) {
    handleUnauthorized(response.status);
    let detail = await response.text();
    try {
      detail = (JSON.parse(detail) as { detail?: string }).detail ?? detail;
    } catch {
      /* a non-JSON error body is still the best message available */
    }
    throw new ApiError(`${method} ${path} -> ${response.status}: ${detail}`, response.status);
  }

  // 204 No Content — DELETE /watchlist/{ticker} is the only one today.
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ── Research ────────────────────────────────────────────

/** Ask a research question and wait for the whole answer. 30–60s is normal. */
export function ask(query: string, threadId?: string): Promise<QueryResponse> {
  return request<QueryResponse>('POST', '/research/query', {
    body: threadId ? { query, thread_id: threadId } : { query },
  });
}

/**
 * Ask a research question, receiving node-by-node progress as it happens.
 *
 * ══ WHY fetch + A MANUAL PARSER, NOT EventSource ══
 *   EventSource is GET-only and cannot send a JSON body, and the endpoint is
 *   a POST carrying the question. So the SSE framing is parsed by hand off
 *   the response stream. The format is the one _sse() writes in
 *   src/api/research_routes.py: "event: <name>\ndata: <json>\n\n".
 *
 *   Frames are dispatched to `onEvent` as they arrive; the promise resolves
 *   with the `final` frame's payload, or rejects on an `error` frame.
 */
export async function askStream(
  query: string,
  onEvent: (event: StreamEvent) => void,
  opts: { threadId?: string; signal?: AbortSignal } = {},
): Promise<QueryResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/research/query/stream`, {
      method: 'POST',
      headers: { ...authHeaders(), 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(opts.threadId ? { query, thread_id: opts.threadId } : { query }),
      signal: opts.signal,
    });
  } catch (exc) {
    if (exc instanceof DOMException && exc.name === 'AbortError') throw exc;
    throw new ApiError(`Could not reach the API at ${API_BASE} — is it running? (${String(exc)})`);
  }

  if (!response.ok || !response.body) {
    handleUnauthorized(response.status);
    throw new ApiError(`POST /research/query/stream -> ${response.status}: ${await response.text()}`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let final: QueryResponse | null = null;

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line. Anything after the last one is
      // a partial frame and stays in the buffer for the next chunk.
      let split = buffer.indexOf('\n\n');
      while (split !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        split = buffer.indexOf('\n\n');

        const name = /^event:\s*(.+)$/m.exec(frame)?.[1]?.trim();
        const raw = /^data:\s*(.*)$/m.exec(frame)?.[1];
        if (!name || raw === undefined) continue;

        let data: unknown;
        try {
          data = JSON.parse(raw);
        } catch {
          continue;
        }

        const event = { event: name, data } as StreamEvent;
        onEvent(event);
        if (event.event === 'final') final = event.data;
        if (event.event === 'error') throw new ApiError(event.data.detail);
      }
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }

  if (!final) throw new ApiError('The stream ended before the answer arrived.');
  return final;
}

/** Replay one run's full audit trail from the checkpointer. */
export function getThread(threadId: string): Promise<ThreadResponse> {
  return request<ThreadResponse>('GET', `/research/threads/${encodeURIComponent(threadId)}`);
}

/** Recent research runs, newest first. */
export function listRuns(limit = 25): Promise<RunSummary[]> {
  return request<RunSummary[]>('GET', '/research/runs', { params: { limit } });
}

// ── Monitor ─────────────────────────────────────────────

/**
 * Run one monitoring cycle now.
 *
 * May come back with `status: "PENDING_APPROVAL"` — a HIGH alert paused the
 * graph before anything was dispatched. See resumeCycle.
 */
export function runCycle(warmup = false, tickers: string[] = []): Promise<CycleRunResponse> {
  return request<CycleRunResponse>('POST', '/monitor/cycles', { body: { warmup, tickers } });
}

/** Decide a paused cycle's pending alerts. Must cover every pending alert id. */
export function resumeCycle(cycleId: string, decisions: Decisions): Promise<CycleRunResponse> {
  return request<CycleRunResponse>('POST', `/monitor/cycles/${encodeURIComponent(cycleId)}/resume`, {
    body: { decisions },
  });
}

/** Recent monitoring cycles, newest first, optionally filtered by status. */
export function listCycles(limit = 25, status?: string): Promise<Cycle[]> {
  return request<Cycle[]>('GET', '/monitor/cycles', { params: { limit, status } });
}

/** One cycle by id. */
export function getCycle(cycleId: string): Promise<Cycle> {
  return request<Cycle>('GET', `/monitor/cycles/${encodeURIComponent(cycleId)}`);
}

/**
 * The HIGH alerts one paused cycle is actually waiting on, in full.
 *
 * Returns [] for a cycle that does not exist, is not paused, or was already
 * resolved — the dashboard cannot tell those apart from this alone, which is
 * fine: either way there is nothing left to decide.
 */
export function getPendingAlerts(cycleId: string): Promise<Alert[]> {
  return request<Alert[]>('GET', `/monitor/cycles/${encodeURIComponent(cycleId)}/pending`);
}

/** Fired alerts, newest first, optionally filtered. */
export function listAlerts(
  opts: { limit?: number; ticker?: string; severity?: string; status?: string } = {},
): Promise<Alert[]> {
  return request<Alert[]>('GET', '/monitor/alerts', {
    params: { limit: opts.limit ?? 50, ticker: opts.ticker, severity: opts.severity, status: opts.status },
  });
}

/** Every dedup decision, with the similarity score that produced it. */
export function listDecisions(limit = 100, decision?: string): Promise<DedupDecision[]> {
  return request<DedupDecision[]>('GET', '/monitor/decisions', { params: { limit, decision } });
}

// ── Watchlist ───────────────────────────────────────────

/** Watched tickers, with each monitor's last-checked watermark. */
export function getWatchlist(includeInactive = false): Promise<WatchItem[]> {
  return request<WatchItem[]>('GET', '/watchlist', { params: { include_inactive: includeInactive } });
}

/** Add (or reactivate) a watched ticker. Company name is resolved when blank. */
export function addTicker(ticker: string, companyName = ''): Promise<WatchItem> {
  return request<WatchItem>('POST', '/watchlist', { body: { ticker, company_name: companyName } });
}

/** Stop watching a ticker (soft delete). */
export async function removeTicker(ticker: string): Promise<true> {
  await request<void>('DELETE', `/watchlist/${encodeURIComponent(ticker)}`);
  return true;
}

// ── Auth ────────────────────────────────────────────────

/**
 * Whether this deployment requires a sign-in, and against which client.
 *
 * Public, and the one call that must work before a token exists — everything
 * else in this file is behind the guard this answer describes.
 */
export function authConfig(): Promise<AuthConfig> {
  return request<AuthConfig>('GET', '/auth/config');
}

// ── Admin ───────────────────────────────────────────────

/** Liveness and the state of every dependency, including the scheduler. */
export function health(): Promise<Health> {
  return request<Health>('GET', '/health');
}

/** Today's usage against each provider's daily allowance. */
export function budgets(): Promise<Budget[]> {
  return request<Budget[]>('GET', '/admin/budgets');
}

/** Effective configuration, secrets excluded. */
export function config(): Promise<Config> {
  return request<Config>('GET', '/admin/config');
}
