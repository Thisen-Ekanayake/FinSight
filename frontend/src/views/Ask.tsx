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
import type { QueryResponse, RunSummary, StreamEvent } from '../api/types';
import { useResource } from '../hooks/useResource';
import { ErrorNote, Kicker, PageHead } from '../components/primitives';
import { renderAnswer, unlocatedClaims } from '../lib/answer';
import { percent, plural, seconds, stamp } from '../lib/format';

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

export function Ask() {
  const [question, setQuestion] = useState('');
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState<Stage[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const startedRef = useRef(0);

  const runs = useResource(() => api.listRuns(8), []);

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
      if (query.length < 3 || running) return;

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
      } catch (exc) {
        if (!(exc instanceof DOMException && exc.name === 'AbortError')) {
          setError(exc instanceof Error ? exc.message : String(exc));
        }
      } finally {
        setRunning(false);
      }
    },
    [question, running, runs],
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

      <form onSubmit={submit} style={{ display: 'flex', gap: 10, marginBottom: 26 }}>
        <input
          className="input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="How did Apple's gross margin trend over the last three fiscal years?"
          aria-label="Research question"
          style={{ flex: '1 1 auto', minWidth: 0, fontSize: 15 }}
        />
        <button
          className="btn btn-primary"
          type="submit"
          disabled={running || question.trim().length < 3}
          style={{ flex: 'none', padding: '9px 20px' }}
        >
          {running ? 'Working…' : 'Ask'}
        </button>
      </form>

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
