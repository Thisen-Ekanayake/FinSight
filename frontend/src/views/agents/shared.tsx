// ═══════════════════════════════════════════════════════
// FinSight web — shared pieces of the agent labs
// ═══════════════════════════════════════════════════════
//
// useLiveRun   the run list for one agent plus the selected run, polled at
//              700 ms while it trains (progress lives in the database, so a
//              reload or a second tab sees the same bar).
// LivePanel    progress bar + learning curve for whatever is training.
// StatusChip   QUEUED / RUNNING / DONE … as a chip, pulsing while live.
// ═══════════════════════════════════════════════════════

import { useCallback, useEffect, useState, type ReactNode } from 'react';
import * as api from '../../api/client';
import type { PolicyRun } from '../../api/types';
import { Card, Icon, ProgressBar } from '../../components/ui';
import { useResource } from '../../hooks/useResource';
import { pct } from '../../lib/money';
import { TrendChart, type TrendSeries } from '../../viz/finance';

export function StatusChip({ status }: { status: string }) {
  const cls =
    status === 'DONE' || status === 'SUBMITTED'
      ? 'chip-pos'
      : status === 'FAILED' || status === 'REJECTED'
        ? 'chip-neg'
        : status === 'RUNNING' || status === 'QUEUED' || status === 'PENDING' || status === 'SUBMITTING'
          ? 'chip-accent'
          : '';
  const live = status === 'RUNNING' || status === 'QUEUED' || status === 'SUBMITTING';
  return (
    <span className={`chip ${cls}`}>
      {live ? (
        <span style={{ width: 7, height: 7, borderRadius: 9, background: 'currentColor', animation: 'fs-pulse 1s infinite' }} />
      ) : null}
      {status.charAt(0) + status.slice(1).toLowerCase()}
    </span>
  );
}

export function isLive(run: { status: string } | null | undefined): boolean {
  return run?.status === 'QUEUED' || run?.status === 'RUNNING';
}

export function useLiveRun<R>(agent: string) {
  const runs = useResource(() => api.listPolicyRuns<R>(15, agent), [agent]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [active, setActive] = useState<PolicyRun<NonNullable<R>> | null>(null);

  useEffect(() => {
    if (!activeId && runs.data?.length) setActiveId(runs.data[0].run_id);
  }, [runs.data, activeId]);

  useEffect(() => {
    if (!activeId) return;
    let live = true;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const r = await api.getPolicyRun<R>(activeId);
        if (!live) return;
        setActive(r);
        if (isLive(r)) timer = window.setTimeout(tick, 700);
        else runs.reload();
      } catch {
        if (live) timer = window.setTimeout(tick, 2000);
      }
    };
    void tick();
    return () => {
      live = false;
      if (timer) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  const started = useCallback(
    (run: PolicyRun<NonNullable<R>>) => {
      setActive(run);
      setActiveId(run.run_id);
      runs.reload();
    },
    [runs],
  );

  return { runs, active, activeId, setActiveId, started, training: isLive(active) };
}

export function LivePanel<R>({
  run,
  emptyIcon,
  empty,
  curveLabel,
  extraSeries,
  chips,
  unit = 'episodes',
}: {
  run: PolicyRun<R> | null;
  emptyIcon: string;
  empty: ReactNode;
  curveLabel: string;
  extraSeries?: TrendSeries[];
  chips?: ReactNode;
  unit?: string;
}) {
  const training = isLive(run);
  return (
    <Card className="span-2" title="Live training" action={run ? <StatusChip status={run.status} /> : null}>
      {!run ? (
        <div style={{ display: 'grid', placeItems: 'center', minHeight: 300, textAlign: 'center', color: 'var(--ink-52)' }}>
          <div>
            <Icon name={emptyIcon} size={42} style={{ color: 'var(--color-accent)' }} />
            <p style={{ marginTop: 12, maxWidth: '46ch' }}>{empty}</p>
          </div>
        </div>
      ) : (
        <>
          {chips ? (
            <div className="row wrap-row" style={{ gap: 8, marginBottom: 12 }}>
              {chips}
            </div>
          ) : null}
          <ProgressBar
            value={run.progress}
            striped={training}
            height={12}
            tone={run.status === 'FAILED' ? 'var(--neg)' : run.status === 'DONE' ? 'var(--pos)' : undefined}
            label={<span style={{ fontWeight: 500 }}>{run.stage || run.status}</span>}
            right={`${run.episodes_done.toLocaleString()} / ${run.episodes_total.toLocaleString()} ${unit} · ${pct(run.progress, 0)}`}
          />
          {run.error ? (
            <p className="chip chip-neg" style={{ marginTop: 12, whiteSpace: 'normal' }}>
              {run.error}
            </p>
          ) : null}
          <div style={{ marginTop: 18 }}>
            <div className="kicker" style={{ marginBottom: 6 }}>
              {curveLabel}
            </div>
            <TrendChart
              labels={run.curve.map((_, i) => `Batch ${i + 1}`)}
              series={[{ name: 'Agent', values: run.curve, color: 'var(--series-1)', area: true }, ...(extraSeries ?? [])]}
              format={(n) => n.toFixed(3)}
              height={220}
            />
          </div>
        </>
      )}
    </Card>
  );
}

export function RunHistory<R>({
  runs,
  activeId,
  onSelect,
  columns,
}: {
  runs: PolicyRun<R>[] | null;
  activeId: string | null;
  onSelect: (id: string) => void;
  columns: { label: string; render: (r: PolicyRun<R>) => ReactNode; right?: boolean }[];
}) {
  return (
    <Card title="Run history" style={{ marginBottom: 18 }}>
      {runs?.length ? (
        <div style={{ overflowX: 'auto' }}>
          <table className="table">
            <thead>
              <tr>
                <th>Started</th>
                <th>Status</th>
                {columns.map((c) => (
                  <th key={c.label} className={c.right ? 'r' : undefined}>
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr
                  key={r.run_id}
                  onClick={() => onSelect(r.run_id)}
                  style={{ cursor: 'pointer', background: r.run_id === activeId ? 'var(--color-accent-100)' : undefined }}
                >
                  <td className="num">{r.created_at ? new Date(r.created_at).toLocaleString() : '—'}</td>
                  <td>
                    <StatusChip status={r.status} />
                  </td>
                  {columns.map((c) => (
                    <td key={c.label} className={c.right ? 'r num' : undefined}>
                      {c.render(r)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="text-muted" style={{ margin: 0 }}>
          No runs yet.
        </p>
      )}
    </Card>
  );
}
