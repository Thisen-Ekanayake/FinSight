// ═══════════════════════════════════════════════════════
// FinSight web — Ask
// ═══════════════════════════════════════════════════════
//
// Purpose : One question, one answer, with every number traceable.
//
// ══ THE PROGRESS IS REAL ══
//   A research question takes 30–60 seconds. That is too long to hide behind
//   a spinner, so this view uses POST /research/query/stream and shows the
//   specialists arriving one at a time — the node frames are the actual
//   supersteps the graph executed. The bar is the only estimated thing on the
//   screen, and it is capped below 100% until the `final` frame lands so it
//   never claims to be finished before it is.
//
// ══ THE FAN-OUT IS THE POINT ══
//   Repeated node names in the stream are one Send per ticker inside a single
//   superstep. They are counted rather than deduplicated, because that
//   repetition is the evidence the fan-out was parallel.
// ═══════════════════════════════════════════════════════

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as api from '../api/client';
import type { Quota, QueryResponse, QuotaExceeded, RunSummary, SeriesPoint, StreamEvent } from '../api/types';
import { useResource } from '../hooks/useResource';
import { ContactSales } from '../components/ContactSales';
import { ErrorNote, Kicker, PageHead } from '../components/primitives';
import { renderAnswer, unlocatedClaims } from '../lib/answer';
import { compact, humanise, percent, plural, seconds, stamp } from '../lib/format';
import type { PlotPoint } from '../viz/plot';
import { BarChart, LineChart, PlotLegend, PlotSource } from '../viz/plot';

/**
 * What each graph node is doing, in the words of someone who does not have
 * the source open. Unknown names fall back to the raw node id rather than
 * being hidden — a node that appears here and has no entry is a graph change
 * the UI has not caught up with, and silence would hide it.
 */
const NODE_LABELS: Record<string, string> = {
  router: 'reading the question',
  fundamentals: 'pulling filed financials',
  filings_rag: 'reading filings from the index',
  macro: 'pulling macro series',
  technical: 'pulling price action',
  aggregator: 'merging findings, resolving conflicts',
  citation_verifier: 'checking every number against a tool result',
  finalize: 'finishing',
};

/** Rough share of a run each stage represents — the progress bar's only guess. */
const NODE_WEIGHT: Record<string, number> = {
  router: 0.08,
  fundamentals: 0.22,
  filings_rag: 0.22,
  macro: 0.12,
  technical: 0.12,
  aggregator: 0.14,
  citation_verifier: 0.2,
  finalize: 0.04,
};

interface Stage {
  node: string;
  at: number;
  findings: number;
  errors: string[];
}

export function Ask({ quota, onSpend }: { quota?: Quota | null; onSpend?: () => void }) {
  const [question, setQuestion] = useState('');
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState<Stage[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Set from a 402. The server is authoritative: a stale client count — a
  // second tab, another device — must still produce the CTA rather than a
  // raw error string, so this is what actually gates the form once it is set.
  const [blocked, setBlocked] = useState<QuotaExceeded | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const startedRef = useRef(0);

  const runs = useResource(() => api.listRuns(8), []);

  const metered = quota?.metered ?? false;
  const exhausted = blocked !== null || (metered && quota!.remaining <= 0);

  // A wall clock rather than an animation: the number on screen is how long
  // the person has actually been waiting.
  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setElapsed((performance.now() - startedRef.current) / 1000), 100);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const submit = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault();
      const query = question.trim();
      if (query.length < 3 || running || exhausted) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setRunning(true);
      setStages([]);
      setResult(null);
      setError(null);
      setElapsed(0);
      startedRef.current = performance.now();

      const onEvent = (frame: StreamEvent) => {
        if (frame.event !== 'node') return;
        setStages((prev) => [
          ...prev,
          {
            node: frame.data.node,
            at: (performance.now() - startedRef.current) / 1000,
            findings: frame.data.findings,
            errors: frame.data.errors ?? [],
          },
        ]);
      };

      try {
        const final = await api.askStream(query, onEvent, { signal: controller.signal });
        setResult(final);
        runs.reload();
        // One extra cheap GET per query, rather than threading the new count
        // back through the SSE `final` frame or a custom response header —
        // the latter would need expose_headers on a split-origin deployment.
        onSpend?.();
      } catch (exc) {
        if (exc instanceof api.QuotaError) {
          setBlocked(exc.quota);
          onSpend?.();
        } else if (!(exc instanceof DOMException && exc.name === 'AbortError')) {
          setError(exc instanceof Error ? exc.message : String(exc));
        }
      } finally {
        setRunning(false);
      }
    },
    [question, running, exhausted, runs, onSpend],
  );

  const progress = useMemo(() => {
    if (result) return 1;
    const done = stages.reduce((total, stage) => total + (NODE_WEIGHT[stage.node] ?? 0.1), 0);
    // Never let an estimate claim completion; the `final` frame does that.
    return Math.min(0.94, done);
  }, [stages, result]);

  const current = stages.length > 0 ? stages[stages.length - 1] : null;
  const stageLabel = result
    ? `finished · ${percent(result.verification.citation_coverage)} of claims traced` +
      (result.verification.unsupported_claims.length > 0
        ? `, ${result.verification.unsupported_claims.length} flagged`
        : '')
    : running
      ? `${NODE_LABELS[current?.node ?? 'router'] ?? current?.node ?? 'starting'}…`
      : 'ask something to begin';

  return (
    <main style={{ maxWidth: 840, margin: '0 auto', padding: '72px 40px 120px' }}>
      <PageHead title="Ask a question">
        Specialists read the primary documents. Every number is checked against a figure a tool actually returned —
        anything that cannot be traced is marked, not deleted.
      </PageHead>

      <form onSubmit={submit} style={{ display: 'flex', gap: 10, marginBottom: metered ? 10 : 26 }}>
        <input
          className="input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="How did Apple's gross margin trend over the last three fiscal years?"
          aria-label="Research question"
          disabled={exhausted}
          style={{ flex: '1 1 auto', minWidth: 0, fontSize: 15 }}
        />
        <button
          className="btn btn-primary"
          type="submit"
          disabled={running || exhausted || question.trim().length < 3}
          style={{ flex: 'none', padding: '9px 20px' }}
        >
          {running ? 'Working…' : 'Ask'}
        </button>
      </form>

      {/* Shown before a query is spent, not after — the last free query and
          the first refused one are otherwise indistinguishable until it is
          too late to choose a different question. */}
      {metered && !exhausted ? (
        <p
          className="mono"
          style={{ fontSize: 12, letterSpacing: '0.04em', color: 'var(--ink-58)', margin: '0 0 26px' }}
        >
          {quota!.remaining} of {quota!.limit} free {quota!.remaining === 1 ? 'query' : 'queries'} left
        </p>
      ) : null}

      {exhausted ? (
        <div style={{ marginBottom: 34 }}>
          <ContactSales
            contactUrl={blocked?.contact_url ?? quota?.contact_url ?? ''}
            used={blocked?.used ?? quota?.used}
            limit={blocked?.limit ?? quota?.limit}
          />
        </div>
      ) : null}

      <div
        className="mono"
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 14,
          fontSize: 12.5,
          letterSpacing: '0.04em',
          color: 'var(--ink-58)',
          marginBottom: 8,
        }}
      >
        <span>{stageLabel}</span>
        <span style={{ marginLeft: 'auto' }}>
          {result ? seconds(result.latency_ms) : running ? `${elapsed.toFixed(1)}s` : '—'}
        </span>
      </div>
      <div style={{ height: 2, background: 'var(--color-divider)', marginBottom: 34 }}>
        <div
          style={{
            height: 2,
            width: `${(progress * 100).toFixed(1)}%`,
            background: 'var(--color-accent)',
            transition: 'width 0.5s ease',
          }}
        />
      </div>

      {stages.length > 0 && !result ? (
        <div style={{ marginBottom: 52 }}>
          {stages.map((stage, i) => (
            <div
              key={`${stage.node}-${i}`}
              className="rise"
              style={{
                display: 'flex',
                gap: 14,
                alignItems: 'baseline',
                padding: '7px 0',
                borderTop: '1px solid var(--color-divider)',
                fontSize: 13.5,
              }}
            >
              <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-45)', flex: '0 0 52px' }}>
                {stage.at.toFixed(1)}s
              </span>
              <span style={{ flex: 1, minWidth: 0 }}>{NODE_LABELS[stage.node] ?? stage.node}</span>
              <span className="mono" style={{ fontSize: 11.5, color: stage.errors.length ? 'var(--alert-ink)' : 'var(--ink-45)' }}>
                {stage.errors.length > 0 ? 'failed' : stage.findings > 0 ? `${stage.findings} findings` : ''}
              </span>
            </div>
          ))}
        </div>
      ) : null}

      <ErrorNote error={error} />

      {result ? <Answer result={result} /> : null}

      {!result && !running && (runs.data?.length ?? 0) > 0 ? <RecentRuns runs={runs.data ?? []} /> : null}
    </main>
  );
}

function Answer({ result }: { result: QueryResponse }) {
  const { verification, citations } = result;
  const blocks = useMemo(
    () => renderAnswer(result.answer, citations, verification.unsupported_claims),
    [result.answer, citations, verification.unsupported_claims],
  );
  const orphans = useMemo(
    () => unlocatedClaims(result.answer, verification.unsupported_claims),
    [result.answer, verification.unsupported_claims],
  );

  return (
    <section className="rise">
      <h2 style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 28, margin: '0 0 18px', letterSpacing: '-0.01em' }}>
        {result.query}
      </h2>

      {blocks.map((block, i) =>
        block.kind === 'bullet' ? (
          <p
            key={i}
            style={{
              fontSize: 17,
              lineHeight: 1.65,
              margin: '0 0 10px',
              paddingLeft: 20,
              maxWidth: '64ch',
              textWrap: 'pretty',
              position: 'relative',
            }}
          >
            <span style={{ position: 'absolute', left: 4, color: 'var(--ink-45)' }}>·</span>
            {block.content}
          </p>
        ) : (
          <p key={i} style={{ fontSize: 17, lineHeight: 1.65, margin: '0 0 16px', maxWidth: '64ch', textWrap: 'pretty' }}>
            {block.content}
          </p>
        ),
      )}

      {verification.unsupported_claims.length > 0 ? (
        <div
          className="mono"
          style={{
            fontSize: 12.5,
            lineHeight: 1.7,
            letterSpacing: '0.03em',
            margin: '20px 0 34px',
            padding: '0 0 0 16px',
            borderLeft: '2px solid var(--alert)',
            color: 'var(--alert-ink)',
            maxWidth: '60ch',
          }}
        >
          <p style={{ margin: 0 }}>
            {verification.unsupported_claims.length === 1
              ? 'One claim is not supported by any value a tool returned.'
              : `${verification.unsupported_claims.length} claims are not supported by any value a tool returned.`}{' '}
            Marked, not removed — judge them yourself.
          </p>
          {orphans.length > 0 ? (
            <ul style={{ margin: '10px 0 0', paddingLeft: 18 }}>
              {orphans.map((claim, i) => (
                <li key={i} style={{ marginBottom: 4 }}>
                  {claim.claim} <span style={{ opacity: 0.7 }}>— {claim.reason}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : (
        <div style={{ marginBottom: 34 }} />
      )}

      <SeriesCharts series={result.series} />

      <div
        style={{
          display: 'flex',
          gap: 40,
          flexWrap: 'wrap',
          padding: '20px 0',
          borderTop: '1px solid var(--color-divider)',
          borderBottom: '1px solid var(--color-divider)',
          marginBottom: 34,
        }}
      >
        <Fact value={percent(verification.citation_coverage)} label="claims traced" />
        <Fact
          value={String(verification.verified_count)}
          label={verification.verified_count === 1 ? 'claim verified' : 'claims verified'}
        />
        <Fact value={String(citations.length)} label={citations.length === 1 ? 'source read' : 'sources read'} />
        <Fact
          value={String(verification.unsupported_claims.length)}
          label={verification.unsupported_claims.length === 1 ? 'claim flagged' : 'claims flagged'}
        />
        {result.repair_count > 0 ? <Fact value={String(result.repair_count)} label="repair passes" /> : null}
      </div>

      {result.conflicts.length > 0 ? (
        <>
          <Kicker>Sources that disagreed</Kicker>
          {result.conflicts.map((conflict, i) => (
            <div key={i} style={{ padding: '11px 2px', borderBottom: '1px solid var(--color-divider)', fontSize: 14 }}>
              <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 600 }}>
                {conflict.metric}
                {conflict.ticker ? ` · ${conflict.ticker}` : ''}
              </span>
              <p style={{ margin: '4px 0 0', color: 'var(--ink-62)', fontSize: 13.5 }}>
                {conflict.values.map(([source, value]) => `${source} said ${value}`).join(', ')} — took{' '}
                {conflict.chosen_source} ({conflict.chosen_value}), a {percent(conflict.rel_difference, 1)} spread.
              </p>
            </div>
          ))}
          <div style={{ height: 26 }} />
        </>
      ) : null}

      <Kicker>Read from</Kicker>
      {citations.length === 0 ? (
        <p style={{ fontSize: 14, color: 'var(--ink-58)' }}>No sources were attached to this answer.</p>
      ) : (
        citations.map((cite, i) => (
          <a
            key={`${cite.source_id}-${i}`}
            href={cite.url}
            target="_blank"
            rel="noreferrer noopener"
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              gap: 16,
              padding: '11px 2px',
              borderBottom: '1px solid var(--color-divider)',
              color: 'inherit',
              fontSize: 14,
            }}
          >
            <span>
              <span className="mono" style={{ color: 'var(--color-accent)', marginRight: 8 }}>
                {i + 1}
              </span>
              {cite.source_type} · {cite.source_id}
            </span>
            <span className="mono" style={{ fontSize: 12, color: 'var(--ink-45)', flex: 'none' }}>
              {cite.as_of}
            </span>
          </a>
        ))
      )}

      {result.errors.length > 0 ? (
        <p style={{ fontSize: 13.5, lineHeight: 1.6, margin: '26px 0 0', maxWidth: '62ch', color: 'var(--ink-62)', textWrap: 'pretty' }}>
          This answer is partial. {result.errors.join(' ')} The rest of the specialists ran normally — a spent daily
          allowance is the usual cause, not a fault.
        </p>
      ) : null}

      <p className="mono" style={{ fontSize: 11.5, margin: '26px 0 0', color: 'var(--ink-45)' }}>
        {result.agents_used.join(' · ')} · {plural(result.branch_count, 'branch', 'branches')} · thread{' '}
        {result.thread_id}
      </p>
    </section>
  );
}

interface MetricChart {
  metric: string;
  unit: string | null;
  periods: string[];
  byTicker: { ticker: string; points: PlotPoint[] }[];
  providers: string[];
  pointCount: number;
}

/** "FY2025", "2025 FY" and "2025-09-27" all order by the year they name. */
function byPeriod(a: string, b: string): number {
  const ya = Number(a.match(/\d{4}/)?.[0] ?? Number.NaN);
  const yb = Number(b.match(/\d{4}/)?.[0] ?? Number.NaN);
  if (Number.isFinite(ya) && Number.isFinite(yb) && ya !== yb) return ya - yb;
  return a.localeCompare(b);
}

/**
 * Group a run's points into one chart per metric.
 *
 * Per metric rather than per ticker, because revenue and gross margin share no
 * axis: one is dollars in the hundreds of billions, the other a percentage
 * under 100. On a common axis the percentage becomes a flat line along the
 * floor, which looks like a finding and is an artefact of the scale.
 */
function groupSeries(series: SeriesPoint[]): MetricChart[] {
  const byMetric = new Map<string, SeriesPoint[]>();
  for (const point of series) {
    const bucket = byMetric.get(point.metric);
    if (bucket) bucket.push(point);
    else byMetric.set(point.metric, [point]);
  }

  const charts: MetricChart[] = [];

  for (const [metric, points] of byMetric) {
    const periods = [...new Set(points.map((p) => p.period))].sort(byPeriod);
    const tickers = [...new Set(points.map((p) => p.ticker))].sort();

    charts.push({
      metric,
      unit: points[0]?.unit ?? null,
      periods,
      byTicker: tickers.map((ticker) => ({
        ticker,
        points: periods.map((period) => {
          const hit = points.find((p) => p.ticker === ticker && p.period === period);
          // NaN, not 0, for a period a company did not report. Zero is a
          // value, and a missing filing drawn at zero is a fabricated cliff.
          return { label: period, value: hit ? hit.value : Number.NaN };
        }),
      })),
      providers: [...new Set(points.map((p) => p.provider).filter((p): p is string => Boolean(p)))],
      pointCount: points.length,
    });
  }

  return charts.sort((a, b) => a.metric.localeCompare(b.metric));
}

/** Axis labels in the metric's own unit — a percentage is not a dollar. */
function axisFormat(unit: string | null): (value: number) => string {
  const u = unit?.toLowerCase() ?? '';
  if (u.includes('percent') || u === '%') return (v) => `${compact(v)}%`;
  if (u === 'usd' || u.includes('dollar')) return (v) => `$${compact(v)}`;
  return compact;
}

/**
 * The filed figures behind the answer.
 *
 * ══ WHY A CHART CAN BE ABSENT ══
 *   Most questions produce no series at all — only the fundamentals
 *   specialist reports figures that belong to reporting periods, and a
 *   question about press coverage or price action has nothing to put on an
 *   axis. This renders nothing rather than an empty frame, because a blank
 *   chart reads as a failure when it is simply the wrong shape of question.
 *
 *   A single point is also dropped. One reading is not a trend, and drawing
 *   an axis around a lone value dresses a number the prose already gave as
 *   though it were an analysis.
 */
function SeriesCharts({ series }: { series: SeriesPoint[] }) {
  const charts = useMemo(() => groupSeries(series).filter((c) => c.pointCount > 1), [series]);
  if (charts.length === 0) return null;

  return (
    <section style={{ marginBottom: 34 }}>
      <Kicker>The filed figures</Kicker>
      <div style={{ display: 'grid', gap: 30 }}>
        {charts.map((chart) => (
          <MetricPlot key={chart.metric} chart={chart} />
        ))}
      </div>
    </section>
  );
}

function MetricPlot({ chart }: { chart: MetricChart }) {
  const format = axisFormat(chart.unit);
  const trend = chart.periods.length > 1;
  const names = chart.byTicker.map((s) => s.ticker);
  const title = `${humanise(chart.metric)} by ${trend ? 'reporting period' : 'ticker'}`;
  const span = trend ? `${chart.periods[0]}–${chart.periods[chart.periods.length - 1]}` : chart.periods[0];

  return (
    <div>
      <p
        className="mono"
        style={{ fontSize: 12, letterSpacing: '0.06em', color: 'var(--ink-70)', margin: '0 0 10px' }}
      >
        {humanise(chart.metric)}
        {chart.unit ? ` · ${chart.unit}` : ''}
      </p>

      {trend ? (
        <>
          <LineChart
            series={chart.byTicker.map((s) => ({ name: s.ticker, points: s.points }))}
            format={format}
            title={title}
          />
          <PlotLegend names={names} />
        </>
      ) : (
        <BarChart
          points={chart.byTicker.map((s) => ({ label: s.ticker, value: s.points[0]?.value ?? Number.NaN }))}
          format={format}
          title={title}
        />
      )}

      {/* Provenance, not decoration: the fundamentals chain falls through
          EDGAR -> yfinance -> FMP, and a fallback estimate must not be
          presented with the authority of a filed figure. */}
      <PlotSource>
        {span} · {chart.providers.length > 0 ? chart.providers.join(', ') : 'source not recorded'}
      </PlotSource>
    </div>
  );
}

function Fact({ value, label }: { value: string; label: string }) {
  return (
    <div>
      <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 26, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 11.5, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-52)', marginTop: 5 }}>
        {label}
      </div>
    </div>
  );
}

function RecentRuns({ runs }: { runs: RunSummary[] }) {
  return (
    <section>
      <Kicker>Asked before</Kicker>
      {runs.map((run) => (
        <div
          key={run.thread_id}
          style={{
            display: 'flex',
            gap: 16,
            alignItems: 'baseline',
            padding: '12px 2px',
            borderTop: '1px solid var(--color-divider)',
            fontSize: 14,
          }}
        >
          <span style={{ flex: 1, minWidth: 0, textWrap: 'pretty' }}>{run.query}</span>
          <span
            className="mono"
            style={{ fontSize: 11.5, flex: 'none', color: run.verification_passed ? 'var(--ink-52)' : 'var(--alert-ink)' }}
          >
            {run.verification_passed
              ? `${percent(run.citation_coverage)} traced`
              : `check failed — ${percent(run.citation_coverage)}`}
          </span>
          <span className="mono" style={{ fontSize: 11.5, flex: 'none', color: 'var(--ink-45)' }}>
            {stamp(run.created_at)}
          </span>
        </div>
      ))}
    </section>
  );
}
