// ═══════════════════════════════════════════════════════
// FinSight web — Findings
// ═══════════════════════════════════════════════════════
//
// Purpose : Everything the watching half reported, newest first, and — under
//           each item — what it chose NOT to report.
//
// ══ THE DEDUP LOG IS THE POINT, NOT AN APPENDIX ══
//   Recognising three headlines as one event is the most interesting thing
//   this subsystem does, and it was rendered as a table of decimals. Here the
//   suppressed candidates hang under the alert they collapsed into, in their
//   own words, with the score as a trailing detail rather than the subject.
//   A dedup engine whose decisions are invisible is indistinguishable from
//   one dropping alerts through a bug — which is exactly why the backend
//   exposes GET /monitor/decisions in the first place.
//
// ══ THE THRESHOLDS ARE READ, NOT WRITTEN DOWN ══
//   The sentence explaining where the fold line sits uses the live values
//   from /admin/config. They are re-tuned whenever the embedding model
//   changes, and a hardcoded 0.89 in this file would quietly start lying.
// ═══════════════════════════════════════════════════════

import { useMemo, useState } from 'react';
import * as api from '../api/client';
import type { Alert, Cycle, DedupDecision } from '../api/types';
import { useResource } from '../hooks/useResource';
import { Blueprint, Empty, ErrorNote, Kicker, Loading, PageHead, SeverityMark } from '../components/primitives';
import { clock, humanise, plural, seconds, severityInk, stamp } from '../lib/format';
import { BarChart, FunnelBars, PlotSource } from '../viz/plot';

type Filter = 'any' | 'HIGH' | 'MED' | 'LOW';

const DECISION_LABELS: Record<string, string> = {
  FIRE: 'REPORTED',
  SUPPRESS_EXACT: 'FOLDED IN',
  SUPPRESS_SEMANTIC: 'FOLDED IN',
  MERGE: 'MERGED',
  ESCALATE: 'SECOND LOOK',
};

export function Findings({ onCycleFinished }: { onCycleFinished: () => void }) {
  const [filter, setFilter] = useState<Filter>('any');
  const [open, setOpen] = useState<string | null>(null);
  const [runState, setRunState] = useState<{ busy: boolean; note: string | null; error: string | null }>({
    busy: false,
    note: null,
    error: null,
  });

  const alerts = useResource(
    () => api.listAlerts({ limit: 60, severity: filter === 'any' ? undefined : filter }),
    [filter],
  );
  const decisions = useResource(() => api.listDecisions(500), []);
  const cycles = useResource(() => api.listCycles(12), []);
  const config = useResource(() => api.config(), []);

  // Suppressed candidates, indexed by the alert they collapsed into.
  const retellings = useMemo(() => {
    const map = new Map<string, DedupDecision[]>();
    for (const decision of decisions.data ?? []) {
      if (decision.decision === 'FIRE' || !decision.parent_alert_id) continue;
      const list = map.get(decision.parent_alert_id) ?? [];
      list.push(decision);
      map.set(decision.parent_alert_id, list);
    }
    for (const list of map.values()) list.sort((a, b) => b.score - a.score);
    return map;
  }, [decisions.data]);

  const runCycle = async () => {
    setRunState({ busy: true, note: null, error: null });
    try {
      const result = await api.runCycle();
      const held = result.pending_approval.length;
      setRunState({
        busy: false,
        error: null,
        note:
          held > 0
            ? `${plural(held, 'HIGH finding')} stopped the run and ${held === 1 ? 'is' : 'are'} waiting on you — see the Desk. Nothing was sent.`
            : `${result.candidate_count} candidates weighed, ${result.fired.length} reported, ${result.suppressed.length} folded away in ${seconds(result.duration_ms)}.`,
      });
      alerts.reload();
      decisions.reload();
      cycles.reload();
      onCycleFinished();
    } catch (exc) {
      setRunState({ busy: false, note: null, error: exc instanceof Error ? exc.message : String(exc) });
    }
  };

  const tauHigh = config.data?.dedup_tau_high;
  const tauLow = config.data?.dedup_tau_low;

  return (
    <main style={{ maxWidth: 900, margin: '0 auto', padding: '72px 40px 120px' }}>
      <PageHead title="Everything it found">
        Most recent first. Severity comes from fixed rules, not from a model's opinion, so it means the same thing
        every time.
      </PageHead>

      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap', marginBottom: 20 }}>
        <div className="seg" role="radiogroup" aria-label="Severity">
          {(['any', 'HIGH', 'MED', 'LOW'] as Filter[]).map((option) => (
            <label key={option} className="seg-opt">
              <input type="radio" name="fs-sev" checked={filter === option} onChange={() => setFilter(option)} />
              {option === 'any' ? 'Everything' : option}
            </label>
          ))}
        </div>

        <button className="btn btn-secondary" type="button" onClick={runCycle} disabled={runState.busy} style={{ marginLeft: 'auto' }}>
          {runState.busy ? 'Run in progress…' : 'Run one now'}
        </button>
      </div>

      {runState.note ? (
        <p style={{ fontSize: 13.5, color: 'var(--ink-62)', margin: '0 0 20px', maxWidth: '68ch', textWrap: 'pretty' }}>
          {runState.note}
        </p>
      ) : null}
      {runState.busy ? (
        <p className="mono" style={{ fontSize: 12.5, color: 'var(--ink-45)', margin: '0 0 20px' }}>
          Checking filings, news, and prices for every watched ticker. A minute or more is normal.
        </p>
      ) : null}
      <ErrorNote error={runState.error} />
      <ErrorNote error={alerts.error} />

      {alerts.loading ? <Loading /> : null}

      {!alerts.loading && (alerts.data?.length ?? 0) === 0 ? (
        <Empty headline={filter === 'any' ? 'Nothing has been reported yet.' : `No ${filter} findings.`}>
          {filter === 'any'
            ? 'Most cycles report nothing at all — the watching half exists to weigh candidates and throw away the noise. Run one now to see what it finds.'
            : 'Severity is scored by rule, and HIGH in particular is rare by design. Widen the filter to see the rest.'}
        </Empty>
      ) : null}

      {(alerts.data ?? []).map((alert) => (
        <Finding
          key={alert.alert_id}
          alert={alert}
          retellings={retellings.get(alert.alert_id) ?? []}
          open={open === alert.alert_id}
          onToggle={() => setOpen(open === alert.alert_id ? null : alert.alert_id)}
          tauHigh={tauHigh}
          tauLow={tauLow}
        />
      ))}

      <DedupMix decisions={decisions.data ?? []} />

      <Runs cycles={cycles.data} error={cycles.error} />
    </main>
  );
}

function Finding({
  alert,
  retellings,
  open,
  onToggle,
  tauHigh,
  tauLow,
}: {
  alert: Alert;
  retellings: DedupDecision[];
  open: boolean;
  onToggle: () => void;
  tauHigh: number | undefined;
  tauLow: number | undefined;
}) {
  const folded = retellings.length;

  return (
    <article style={{ padding: '26px 0', borderTop: '1px solid var(--color-divider)' }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', flexWrap: 'wrap', marginBottom: 10 }}>
        <SeverityMark severity={alert.severity} />
        <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 15, letterSpacing: '0.06em' }}>
          {alert.ticker || 'MACRO'}
        </span>
        <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-45)', marginLeft: 'auto' }}>
          {clock(alert.fired_at)} · {alert.status.toLowerCase().replace(/_/g, ' ')}
          {alert.occurrence_count > 1 ? ` · seen ${alert.occurrence_count}×` : ''}
        </span>
      </div>

      <p
        style={{
          fontSize: 20,
          lineHeight: 1.32,
          fontFamily: 'var(--font-heading)',
          fontWeight: 600,
          margin: '0 0 10px',
          maxWidth: '52ch',
          textWrap: 'pretty',
        }}
      >
        {alert.headline}
      </p>
      <p style={{ fontSize: 14.5, lineHeight: 1.6, maxWidth: '68ch', margin: '0 0 12px', color: 'var(--ink-68, var(--ink-70))', textWrap: 'pretty' }}>
        {alert.detail}
      </p>

      <div style={{ display: 'flex', gap: 18, alignItems: 'baseline', flexWrap: 'wrap' }}>
        <a
          href="#dedup"
          onClick={(e) => {
            e.preventDefault();
            onToggle();
          }}
          style={{ fontSize: 13 }}
        >
          {open
            ? 'Hide the retellings'
            : folded > 0
              ? `Seen again ${plural(folded, 'time')} — show what was folded in`
              : 'Show why it was reported'}
        </a>
        <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-45)' }}>
          {humanise(alert.alert_type)}
        </span>
      </div>

      {open ? (
        <div style={{ marginTop: 16, paddingLeft: 16, borderLeft: '2px solid var(--color-divider)', maxWidth: '68ch' }}>
          {folded === 0 ? (
            <p style={{ fontSize: 13.5, color: 'var(--ink-58)', margin: '0 0 10px' }}>
              Nothing has been folded into this one. It is the first and so far only telling of this event.
            </p>
          ) : (
            retellings.map((decision, i) => (
              <div key={`${decision.dedup_key}-${i}`} style={{ display: 'flex', gap: 14, alignItems: 'baseline', padding: '7px 0', fontSize: 13.5 }}>
                <span
                  className="mono"
                  style={{
                    fontSize: 11.5,
                    letterSpacing: '0.06em',
                    flex: '0 0 92px',
                    color: decision.decision === 'ESCALATE' ? 'var(--alert-ink)' : 'var(--ink-52)',
                  }}
                >
                  {DECISION_LABELS[decision.decision] ?? decision.decision}
                </span>
                <span style={{ flex: 1, minWidth: 0, textWrap: 'pretty' }}>{decision.candidate_text}</span>
                <span className="mono" style={{ fontSize: 12, flex: 'none', color: 'var(--ink-52)' }}>
                  {decision.score.toFixed(3)}
                </span>
              </div>
            ))
          )}

          {tauHigh !== undefined && tauLow !== undefined ? (
            <p style={{ fontSize: 12.5, margin: '10px 0 0', color: 'var(--ink-52)', textWrap: 'pretty' }}>
              Folded at or above {tauHigh}. Between {tauLow} and {tauHigh} it is merged into this alert as the same
              event carrying new information, rather than dropped.
            </p>
          ) : null}

          {alert.evidence.length > 0 ? (
            <div style={{ marginTop: 14 }}>
              <Kicker style={{ marginBottom: 6 }}>Evidence</Kicker>
              {alert.evidence.map((cite, k) => (
                <a
                  key={`${cite.source_id}-${k}`}
                  href={cite.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  style={{ display: 'block', fontSize: 13, padding: '3px 0' }}
                >
                  {cite.source_type} · {cite.source_id}
                </a>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

/**
 * The cycle ledger.
 *
 * Kept because "was it even running while I was away?" is a real question and
 * nothing else on this screen answers it — a quiet day and a broken scheduler
 * look identical from an empty findings list alone.
 */
/**
 * What the dedup engine decided, in aggregate.
 *
 * The per-alert log below already shows each fold in its own words. This is
 * the shape of the whole log: a system that reports one candidate in five is
 * doing the job the page's header claims for it, and a system that folds
 * nothing is either seeing genuinely distinct events or quietly broken. Both
 * readings need the totals, and neither survives scrolling a list of decimals.
 *
 * Drawn from /monitor/decisions, which is deliberately NOT the severity-filtered
 * alert list — a mix computed from a filtered view would describe the filter.
 */
function DedupMix({ decisions }: { decisions: DedupDecision[] }) {
  const rows = useMemo(() => {
    const counts = new Map<string, number>();
    for (const decision of decisions) {
      const label = DECISION_LABELS[decision.decision] ?? decision.decision;
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    // Reported first, then the folds, then whatever the backend added that
    // this file has no label for — an unknown decision is worth seeing, not
    // worth hiding behind a sort that buries it.
    const order = ['REPORTED', 'FOLDED IN', 'MERGED', 'SECOND LOOK'];
    return [...counts.entries()]
      .sort((a, b) => {
        const ia = order.indexOf(a[0]);
        const ib = order.indexOf(b[0]);
        return (ia < 0 ? order.length : ia) - (ib < 0 ? order.length : ib);
      })
      .map(([label, value]) => ({
        label: label.toLowerCase(),
        value,
        tone: label === 'SECOND LOOK' ? 'var(--alert)' : undefined,
      }));
  }, [decisions]);

  if (rows.length === 0) return null;

  return (
    <section style={{ marginTop: 56 }}>
      <h2 style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 24, margin: '0 0 6px' }}>
        What was weighed
      </h2>
      <p style={{ fontSize: 14, color: 'var(--ink-62)', margin: '0 0 18px', maxWidth: '58ch' }}>
        Every candidate the engine judged, by what it decided. Folding is the expected outcome, not an error — the
        same event reaches us from several sources at once.
      </p>
      <FunnelBars rows={rows} />
      <PlotSource>{plural(decisions.length, 'decision')} across the recent log</PlotSource>
    </section>
  );
}

function Runs({ cycles, error }: { cycles: Cycle[] | null; error: string | null }) {
  if (error) return <ErrorNote error={error} />;
  if (!cycles || cycles.length === 0) return null;

  // Oldest first: a run chart reads left to right in time, and the list below
  // stays newest first because that is the order you read findings in.
  const charted = [...cycles].reverse();
  const totals = cycles.reduce(
    (acc, c) => ({
      seen: acc.seen + c.candidate_count,
      reported: acc.reported + c.fired_count,
      folded: acc.folded + c.suppressed_count + c.merged_count,
      errors: acc.errors + c.error_count,
    }),
    { seen: 0, reported: 0, folded: 0, errors: 0 },
  );

  return (
    <section style={{ marginTop: 56 }}>
      <h2 style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 24, margin: '0 0 6px' }}>Runs</h2>
      <p style={{ fontSize: 14, color: 'var(--ink-62)', margin: '0 0 18px', maxWidth: '58ch' }}>
        One complete pass of the watching machine. Most of what a run sees never becomes a finding, and that is the
        engine working.
      </p>

      {/* Only worth drawing once there are two runs to compare. A single
          column is a number with axes drawn around it. */}
      {charted.length > 1 ? (
        <div style={{ marginBottom: 28 }}>
          <BarChart
            points={charted.map((cycle) => ({ label: clock(cycle.started_at), value: cycle.fired_count }))}
            format={(v) => String(Math.round(v))}
            title="Findings reported per run"
            height={190}
          />
          <PlotSource>
            {totals.seen.toLocaleString()} seen · {totals.reported.toLocaleString()} reported ·{' '}
            {totals.folded.toLocaleString()} folded
            {totals.errors > 0 ? ` · ${plural(totals.errors, 'error')}` : ''} across {plural(cycles.length, 'run')}
          </PlotSource>
        </div>
      ) : null}

      {cycles.map((cycle) => {
        const held = cycle.status === 'PENDING_APPROVAL';
        return (
          <Blueprint key={cycle.cycle_id} style={{ padding: '14px 16px', marginBottom: 12 }}>
            <div style={{ display: 'flex', gap: 14, alignItems: 'baseline', flexWrap: 'wrap' }}>
              <span className="mono" style={{ fontSize: 12, color: held ? 'var(--alert-ink)' : 'var(--ink-52)', flex: '0 0 auto' }}>
                {held ? 'held' : cycle.warmup ? 'warm-up' : 'complete'}
              </span>
              <span style={{ fontSize: 14, flex: '1 1 auto', minWidth: 0 }}>
                {cycle.candidate_count} seen · {cycle.fired_count} reported ·{' '}
                {cycle.suppressed_count + cycle.merged_count} folded
                {cycle.error_count > 0 ? ` · ${plural(cycle.error_count, 'error')}` : ''}
              </span>
              <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-45)', flex: 'none' }}>
                {stamp(cycle.started_at)} · {seconds(cycle.duration_ms)} · {cycle.api_call_count} calls
              </span>
            </div>
            <div style={{ display: 'flex', gap: 3, marginTop: 10, alignItems: 'flex-end', height: 16 }}>
              {Array.from({ length: 24 }, (_, k) => {
                const share = cycle.candidate_count === 0 ? 0 : cycle.fired_count / cycle.candidate_count;
                const lit = k < Math.round(share * 24);
                return (
                  <span
                    key={k}
                    style={{
                      flex: 1,
                      height: lit ? 14 : 5,
                      background: lit ? (held ? 'var(--alert)' : severityInk('MED')) : 'var(--color-divider)',
                    }}
                  />
                );
              })}
            </div>
          </Blueprint>
        );
      })}
    </section>
  );
}
