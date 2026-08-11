// ═══════════════════════════════════════════════════════
// FinSight web — Desk
// ═══════════════════════════════════════════════════════
//
// Purpose : Answer "is there anything I need to do?" in three seconds, and
//           then let the person do it without going anywhere else.
//
// ══ THE APPROVAL GATE LIVES HERE, NOT ON A PAGE ══
//   A HIGH finding stops its run and nothing is dispatched until a human
//   decides. That was the fourth item in a sidebar; it is now the first thing
//   on the first screen, and the decision is made in place.
//
// ══ REJECT IS THE DEFAULT AND STAYS THE DEFAULT ══
//   `decisionsFor` seeds every pending alert to "reject". Submitting an
//   untouched form dispatches nothing. This is not a neutral "please choose"
//   that happens to start empty — the backend requires a decision for every
//   pending id, so something has to be preselected, and the safe direction is
//   the only defensible one. Do not soften this into a disabled submit.
//
// ══ WHY THE QUIET STATE GOT THE SAME EFFORT AS THE BUSY ONE ══
//   Most cycles produce zero HIGH alerts, so "nothing needs you" is what this
//   screen shows most days. It gets the same 64px headline and a real
//   sentence about what the machine did while nobody was looking.
// ═══════════════════════════════════════════════════════

import { useCallback, useMemo, useState } from 'react';
import type { Alert, Cycle, Decisions } from '../api/types';
import * as api from '../api/client';
import { useResource } from '../hooks/useResource';
import type { View } from '../hooks/useRoute';
import type { VizColors } from '../viz/ambient';
import { beaconsFor } from '../viz/beacons';
import { AmbientField } from '../components/AmbientField';
import { Blueprint, ErrorNote, SeverityMark } from '../components/primitives';
import { clock, plural, since, stamp } from '../lib/format';

export interface PendingCycle {
  cycle: Cycle;
  alerts: Alert[];
}

/**
 * Every paused cycle with the alerts it is actually waiting on.
 *
 * A cycle can carry status PENDING_APPROVAL and still have no pending alerts
 * — /monitor/cycles/{id}/pending reads the checkpointer, and a cycle that was
 * resolved elsewhere (the CLI, another tab) returns [] from it while its row
 * is mid-update. Those are dropped rather than rendered as an empty card.
 */
export async function loadPending(): Promise<PendingCycle[]> {
  const paused = await api.listCycles(25, 'PENDING_APPROVAL');
  const withAlerts = await Promise.all(
    paused.map(async (cycle) => ({ cycle, alerts: await api.getPendingAlerts(cycle.cycle_id) })),
  );
  return withAlerts.filter((entry) => entry.alerts.length > 0);
}

function decisionsFor(alerts: Alert[]): Decisions {
  return Object.fromEntries(alerts.map((a) => [a.alert_id, 'reject' as const]));
}

export function Desk({
  pending,
  pendingError,
  reloadPending,
  vizColors,
  navigate,
}: {
  pending: PendingCycle[] | null;
  pendingError: string | null;
  reloadPending: () => void;
  vizColors: VizColors;
  navigate: (view: View) => void;
}) {
  const health = useResource(() => api.health(), [], { pollMs: 30_000 });
  const watchlist = useResource(() => api.getWatchlist(), []);
  const recent = useResource(() => api.listCycles(100), [], { pollMs: 60_000 });

  const cycles = pending ?? [];
  const pendingCount = cycles.reduce((total, entry) => total + entry.alerts.length, 0);
  const hasPending = pendingCount > 0;

  // What the machine did on its own, over the cycles it still remembers.
  const worked = useMemo(() => {
    const rows = recent.data ?? [];
    return {
      runs: rows.length,
      weighed: rows.reduce((n, c) => n + c.candidate_count, 0),
      reported: rows.reduce((n, c) => n + c.fired_count, 0),
      folded: rows.reduce((n, c) => n + c.suppressed_count + c.merged_count, 0),
      lastRun: rows[0]?.started_at ?? null,
    };
  }, [recent.data]);

  const watching = watchlist.data?.length ?? 0;
  const nextRun = health.data?.scheduler_enabled
    ? `next run ${clock(health.data.scheduler_next_run_at)}`
    : 'scheduler off — runs on request';
  const liveLine = [
    `watching ${plural(watching, 'company', 'companies')}`,
    nextRun,
    worked.lastRun ? `last run ${clock(worked.lastRun)}` : 'no runs yet',
  ].join(' · ');

  return (
    <main>
      <section style={{ position: 'relative', overflow: 'hidden', borderBottom: '1px solid var(--color-divider)' }}>
        <AmbientField colors={vizColors} beacons={beaconsFor(pendingCount)} />

        <div style={{ position: 'relative', maxWidth: 1080, margin: '0 auto', padding: '104px 40px 88px' }}>
          <div
            className="mono"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 9,
              fontSize: 11.5,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: 'var(--ink-52)',
              marginBottom: 26,
            }}
          >
            <span
              style={{
                width: 6,
                height: 6,
                flex: 'none',
                borderRadius: '50%',
                background: health.data?.status === 'ok' ? 'var(--color-accent)' : 'var(--alert)',
                animation: 'fs-pulse 2.6s ease-in-out infinite',
              }}
            />
            <span>{health.error ? 'backend unreachable' : liveLine}</span>
          </div>

          {hasPending ? (
            <>
              <h1
                style={{
                  fontFamily: 'var(--font-heading)',
                  fontWeight: 600,
                  fontSize: 64,
                  lineHeight: 1.02,
                  letterSpacing: '-0.025em',
                  margin: '0 0 22px',
                  maxWidth: '22ch',
                }}
              >
                {pendingCount === 1 ? 'One decision is waiting on you.' : `${pendingCount} decisions are waiting on you.`}
              </h1>
              <p style={{ fontSize: 18, maxWidth: '46ch', margin: '0 0 34px', color: 'var(--ink-74)', textWrap: 'pretty' }}>
                Nothing has been sent, and nothing will be until you say so.
              </p>
              <a className="btn btn-primary blueprint" href="#queue" style={{ padding: '11px 22px', fontSize: 15 }}>
                <i className="corner tl" />
                <i className="corner tr" />
                <i className="corner bl" />
                <i className="corner br" />
                Review them
              </a>
            </>
          ) : (
            <>
              <h1
                style={{
                  fontFamily: 'var(--font-heading)',
                  fontWeight: 600,
                  fontSize: 64,
                  lineHeight: 1.02,
                  letterSpacing: '-0.025em',
                  margin: '0 0 22px',
                }}
              >
                Nothing needs you.
              </h1>
              <p style={{ fontSize: 18, maxWidth: '46ch', margin: 0, color: 'var(--ink-74)', textWrap: 'pretty' }}>
                {worked.runs > 0
                  ? `${plural(worked.runs, 'run')}, ${worked.weighed} findings weighed, ${worked.folded} cleared as noise. Quiet is a result.`
                  : 'No monitoring cycle has run yet. Add tickers to the watchlist, then run one from Findings.'}
              </p>
            </>
          )}
        </div>
      </section>

      {hasPending ? (
        <section id="queue" style={{ maxWidth: 1080, margin: '0 auto', padding: '76px 40px 40px' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap', marginBottom: 34 }}>
            <h2 style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 28, margin: 0, letterSpacing: '-0.01em' }}>
              Waiting on a person
            </h2>
            <span style={{ fontSize: 13.5, color: 'var(--ink-52)' }}>
              Reject is preselected — submitting unchanged sends nothing.
            </span>
          </div>

          <ErrorNote error={pendingError} />

          {cycles.map((entry) => (
            <PendingCycleCard key={entry.cycle.cycle_id} entry={entry} onResolved={reloadPending} />
          ))}
        </section>
      ) : null}

      <section style={{ maxWidth: 1080, margin: '0 auto', padding: '32px 40px 110px' }}>
        <ErrorNote error={hasPending ? null : pendingError} />
        <p style={{ fontSize: 14, margin: 0, color: 'var(--ink-52)', textWrap: 'pretty' }}>
          {hasPending ? 'Everything else went out on its own: ' : ''}
          {worked.reported} reported, {worked.folded} retellings folded away over {plural(worked.runs, 'run')}.{' '}
          <a
            href="/findings"
            onClick={(e) => {
              e.preventDefault();
              navigate('findings');
            }}
          >
            See everything it found
          </a>
        </p>
      </section>
    </main>
  );
}

/**
 * One paused cycle and its decisions.
 *
 * Decision state is local to the card because the backend takes them as one
 * atomic dict: resume_cycle rejects a partial or mismatched map rather than
 * applying part of it, so there is no such thing as submitting one alert.
 */
function PendingCycleCard({ entry, onResolved }: { entry: PendingCycle; onResolved: () => void }) {
  const { cycle, alerts } = entry;
  const [decisions, setDecisions] = useState<Decisions>(() => decisionsFor(alerts));
  const [openWhy, setOpenWhy] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const approved = Object.values(decisions).filter((d) => d === 'approve').length;
  const summary =
    approved === 0
      ? alerts.length === 1
        ? 'Set to Reject — nothing will be sent.'
        : 'All set to Reject — nothing will be sent.'
      : approved === alerts.length
        ? `${plural(approved, 'alert')} will be dispatched.`
        : `${plural(approved, 'alert')} will be dispatched; the rest are rejected.`;

  const submit = useCallback(async () => {
    setSubmitting(true);
    setError(null);
    try {
      await api.resumeCycle(cycle.cycle_id, decisions);
      onResolved();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
      setSubmitting(false);
    }
  }, [cycle.cycle_id, decisions, onResolved]);

  return (
    <Blueprint style={{ padding: '26px 28px', marginBottom: 24 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'baseline',
          flexWrap: 'wrap',
          gap: 12,
          marginBottom: 6,
        }}
      >
        <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 17 }}>
          Run at {stamp(cycle.started_at)}
        </span>
        <span className="mono" style={{ fontSize: 12, color: 'var(--alert-ink)' }}>
          held {since(cycle.started_at)}
        </span>
      </div>

      {alerts.map((alert) => {
        const open = openWhy === alert.alert_id;
        return (
          <div key={alert.alert_id} style={{ padding: '22px 0', borderTop: '1px solid var(--color-divider)' }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'baseline', marginBottom: 10, flexWrap: 'wrap' }}>
              <SeverityMark severity={alert.severity} boxed />
              <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 16, letterSpacing: '0.06em' }}>
                {alert.ticker || 'MACRO'}
              </span>
              {alert.occurrence_count > 1 ? (
                <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-45)' }}>
                  seen {alert.occurrence_count}×
                </span>
              ) : null}
            </div>

            <p
              style={{
                fontFamily: 'var(--font-heading)',
                fontWeight: 600,
                fontSize: 23,
                lineHeight: 1.22,
                margin: '0 0 10px',
                maxWidth: '44ch',
                textWrap: 'pretty',
              }}
            >
              {alert.headline}
            </p>
            <p
              style={{
                fontSize: 14.5,
                lineHeight: 1.6,
                maxWidth: '66ch',
                margin: '0 0 16px',
                color: 'var(--ink-70)',
                textWrap: 'pretty',
              }}
            >
              {alert.detail}
            </p>

            <div style={{ display: 'flex', gap: 26, alignItems: 'center', flexWrap: 'wrap' }}>
              <label className="radio" style={{ fontSize: 14 }}>
                <input
                  type="radio"
                  name={`decision-${alert.alert_id}`}
                  checked={decisions[alert.alert_id] !== 'approve'}
                  disabled={submitting}
                  onChange={() => setDecisions((d) => ({ ...d, [alert.alert_id]: 'reject' }))}
                />
                <span className="dot" />
                Reject
              </label>
              <label className="radio" style={{ fontSize: 14 }}>
                <input
                  type="radio"
                  name={`decision-${alert.alert_id}`}
                  checked={decisions[alert.alert_id] === 'approve'}
                  disabled={submitting}
                  onChange={() => setDecisions((d) => ({ ...d, [alert.alert_id]: 'approve' }))}
                />
                <span className="dot" />
                Approve — send it
              </label>
              <a
                href="#why"
                onClick={(e) => {
                  e.preventDefault();
                  setOpenWhy(open ? null : alert.alert_id);
                }}
                style={{ fontSize: 13 }}
              >
                {open ? 'Hide why it stopped' : 'Why it stopped'}
              </a>
            </div>

            {open ? (
              <div
                style={{
                  marginTop: 16,
                  paddingLeft: 16,
                  borderLeft: '2px solid var(--color-divider)',
                  fontSize: 13.5,
                  lineHeight: 1.6,
                  color: 'var(--ink-66)',
                  maxWidth: '66ch',
                }}
              >
                <div
                  className="mono"
                  style={{ fontSize: 11.5, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 6 }}
                >
                  rule: {alert.alert_type} · severity is scored by rule, never by a model
                </div>
                <p style={{ margin: '0 0 10px' }}>
                  Held because HIGH requires a human before dispatch. First seen {stamp(alert.first_seen_at)}.
                </p>
                {alert.evidence.length > 0 ? (
                  <>
                    <div className="kicker" style={{ marginBottom: 6 }}>
                      Evidence
                    </div>
                    {alert.evidence.map((cite, k) => (
                      <div key={`${cite.source_id}-${k}`} style={{ padding: '5px 0', fontSize: 13 }}>
                        <a href={cite.url} target="_blank" rel="noreferrer noopener">
                          {cite.source_type} · {cite.source_id}
                        </a>
                        <span className="mono" style={{ marginLeft: 8, fontSize: 11.5, color: 'var(--ink-45)' }}>
                          {cite.as_of}
                        </span>
                        {cite.excerpt ? (
                          <p style={{ margin: '4px 0 0', color: 'var(--ink-58)', textWrap: 'pretty' }}>{cite.excerpt}</p>
                        ) : null}
                      </div>
                    ))}
                  </>
                ) : (
                  <p style={{ margin: 0, color: 'var(--ink-58)' }}>
                    No evidence records were attached — the rule fired on the candidate's own text alone.
                  </p>
                )}
              </div>
            ) : null}
          </div>
        );
      })}

      <div style={{ paddingTop: 20, borderTop: '1px solid var(--color-divider)' }}>
        <ErrorNote error={error} />
        <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
          <button className="btn btn-primary" type="button" onClick={submit} disabled={submitting} style={{ padding: '9px 18px' }}>
            {submitting ? 'Submitting…' : 'Submit decisions'}
          </button>
          <span style={{ fontSize: 13, color: 'var(--ink-52)' }}>{summary}</span>
        </div>
      </div>
    </Blueprint>
  );
}
