// ═══════════════════════════════════════════════════════
// FinSight web — Alert prioritiser lab
// ═══════════════════════════════════════════════════════
// Recent monitor alerts, ranked by what you have found useful. Thumbs up /
// down train it instantly; your approve / reject decisions on HIGH alerts
// already did. It reorders — it never hides an alert.
// ═══════════════════════════════════════════════════════

import { useState } from 'react';
import * as api from '../../api/client';
import type { PreferenceRow, RankedAlert } from '../../api/types';
import { Card, Icon, ProgressBar, Toggle, useAction } from '../../components/ui';
import { SeverityMark } from '../../components/primitives';
import type { View } from '../../hooks/useRoute';
import { useResource } from '../../hooks/useResource';
import { humanise, since } from '../../lib/format';
import { pct } from '../../lib/money';

const FEATURE_TITLES: Record<string, string> = { alert_type: 'By alert type', severity: 'By severity', ticker: 'By ticker' };

export function AlertLab({ navigate }: { navigate: (v: View) => void }) {
  const [explore, setExplore] = useState(false);
  const [shuffle, setShuffle] = useState(0);
  const ranked = useResource(() => api.rankedAlerts(explore, 40), [explore, shuffle]);
  const prefs = useResource(() => api.alertPreferences(), []);
  const labels = ranked.data?.labels;
  const forbidden = ranked.error?.includes('403');

  const refresh = () => {
    ranked.reload();
    prefs.reload();
  };

  if (forbidden) {
    return (
      <Card title="Alert prioritiser">
        <p className="text-muted" style={{ margin: 0 }}>
          The prioritiser ranks the monitoring desk's alerts, and monitoring is part of the unlimited tier on this deployment.
        </p>
      </Card>
    );
  }

  return (
    <>
      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <Card title="How it learns">
          <div style={{ display: 'grid', gap: 16 }}>
            <p style={{ margin: 0, fontSize: 14, color: 'var(--ink-62)' }}>
              Every HIGH alert already stops for your approval. Approving one says “this kind matters”; rejecting says it doesn't.
              Thumbs on any alert count double. A Thompson-sampling bandit turns that into a ranking.
            </p>
            <ProgressBar value={labels?.from_approvals ?? 0} max={Math.max(labels?.total ?? 0, 1)} tone="var(--series-2)" label="From approval decisions" right={labels?.from_approvals ?? 0} />
            <ProgressBar value={labels?.explicit ?? 0} max={Math.max(labels?.total ?? 0, 1)} tone="var(--series-1)" label="From your 👍 / 👎" right={labels?.explicit ?? 0} />
            <Toggle
              checked={explore}
              onChange={setExplore}
              label={
                <span>
                  <b>Explore</b>
                  <br />
                  <span style={{ fontSize: 12, color: 'var(--ink-52)' }}>Rank by a posterior draw, so uncertain kinds sometimes rise.</span>
                </span>
              }
            />
            {explore ? (
              <button className="btn btn-sm" type="button" onClick={() => setShuffle((n) => n + 1)}>
                <Icon name="sparkle" size={14} /> Draw again
              </button>
            ) : null}
          </div>
        </Card>

        <Card className="span-2" title="Ranked for you" action={<span className="chip">{ranked.data?.alerts.length ?? 0} recent alerts</span>}>
          {ranked.data && ranked.data.alerts.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <p className="text-muted">No alerts yet. Add tickers to the watchlist and run a monitoring cycle.</p>
              <button className="btn btn-sm" type="button" onClick={() => navigate('findings')}>
                Go to Findings
              </button>
            </div>
          ) : (
            <div style={{ display: 'grid', gap: 8, maxHeight: 560, overflowY: 'auto', paddingRight: 4 }}>
              {(ranked.data?.alerts ?? []).map((a, i) => (
                <AlertRow key={a.alert_id} alert={a} rank={i + 1} onChange={refresh} />
              ))}
            </div>
          )}
        </Card>
      </div>

      <Card title="What it has learned" style={{ marginBottom: 18 }}>
        {prefs.data ? (
          <div className="grid grid-3">
            {Object.entries(prefs.data.preferences).map(([feature, rows]) => (
              <div key={feature}>
                <div className="kicker" style={{ marginBottom: 10 }}>
                  {FEATURE_TITLES[feature] ?? feature}
                </div>
                {rows.length ? (
                  <div style={{ display: 'grid', gap: 12 }}>
                    {rows.slice(0, 8).map((r) => (
                      <BandBar key={r.value} row={r} raw={feature === 'ticker'} />
                    ))}
                  </div>
                ) : (
                  <p className="text-muted" style={{ fontSize: 13, margin: 0 }}>
                    Nothing learned yet.
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : null}
        <p style={{ fontSize: 12, color: 'var(--ink-52)', margin: '14px 0 0' }}>
          Dot: how likely this kind is to be useful to you. Band: 90% credible range — wide means “not sure yet”.
        </p>
      </Card>
    </>
  );
}

function AlertRow({ alert, rank, onChange }: { alert: RankedAlert; rank: number; onChange: () => void }) {
  const { busy, run } = useAction();
  const vote = async (useful: boolean) => {
    const next = alert.feedback === useful ? null : useful;
    await run(() => api.alertFeedback(alert.alert_id, next));
    onChange();
  };
  const p = alert.priority;
  return (
    <div className="card pop" style={{ padding: '12px 14px', display: 'grid', gridTemplateColumns: '28px minmax(0,1fr) auto', gap: 12, alignItems: 'center' }}>
      <span className="num" style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, color: 'var(--ink-45)' }}>
        {rank}
      </span>
      <div style={{ minWidth: 0 }}>
        <div className="row wrap-row" style={{ gap: 8, marginBottom: 4 }}>
          <SeverityMark severity={alert.severity} />
          <b>{alert.ticker}</b>
          <span className="chip" style={{ fontSize: 10.5, padding: '1px 7px' }}>
            {humanise(alert.alert_type)}
          </span>
          {alert.status === 'REJECTED' ? <span className="chip chip-neg" style={{ fontSize: 10.5, padding: '1px 7px' }}>rejected</span> : null}
          <span style={{ fontSize: 12, color: 'var(--ink-45)' }}>{since(alert.fired_at)}</span>
        </div>
        <div style={{ fontSize: 13.5, color: 'var(--ink-70)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={alert.headline}>
          {alert.headline || '—'}
        </div>
        <div style={{ marginTop: 6, maxWidth: 360 }}>
          <ProgressBar
            value={p.priority}
            height={5}
            tone={p.priority >= 0.6 ? 'var(--pos)' : p.priority <= 0.4 ? 'var(--neg)' : 'var(--color-accent)'}
            label={<span style={{ fontSize: 11.5, color: 'var(--ink-52)' }}>Priority {pct(p.priority, 0)}</span>}
            right={<span style={{ fontSize: 11.5 }}>± {pct(p.uncertainty, 0)}</span>}
          />
        </div>
      </div>
      <div className="row" style={{ gap: 4 }}>
        <button
          type="button"
          className="icon-btn"
          aria-label={`Useful: ${alert.headline}`}
          aria-pressed={alert.feedback === true}
          disabled={busy}
          onClick={() => vote(true)}
          style={alert.feedback === true ? { background: 'var(--pos-soft)', color: 'var(--pos)' } : undefined}
        >
          👍
        </button>
        <button
          type="button"
          className="icon-btn"
          aria-label={`Not useful: ${alert.headline}`}
          aria-pressed={alert.feedback === false}
          disabled={busy}
          onClick={() => vote(false)}
          style={alert.feedback === false ? { background: 'var(--neg-soft)', color: 'var(--neg)' } : undefined}
        >
          👎
        </button>
      </div>
    </div>
  );
}

function BandBar({ row, raw }: { row: PreferenceRow; raw: boolean }) {
  const tone = row.mean >= 0.6 ? 'var(--pos)' : row.mean <= 0.4 ? 'var(--neg)' : 'var(--color-accent)';
  return (
    <div>
      <div className="row between" style={{ fontSize: 12.5, marginBottom: 4 }}>
        <span>{raw ? row.value : humanise(row.value)}</span>
        <span className="num" style={{ color: 'var(--ink-58)' }}>
          {pct(row.mean, 0)} · {row.useful}👍 {row.not_useful}👎
        </span>
      </div>
      <div style={{ position: 'relative', height: 8, background: 'var(--color-field)', borderRadius: 8 }}>
        <span style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 1, background: 'var(--ink-42)' }} />
        <span
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: `${row.low * 100}%`,
            width: `${(row.high - row.low) * 100}%`,
            background: `color-mix(in srgb, ${tone} 28%, transparent)`,
            borderRadius: 8,
            transition: 'all 0.6s ease',
          }}
        />
        <span
          style={{
            position: 'absolute',
            top: -2,
            width: 12,
            height: 12,
            marginLeft: -6,
            left: `${row.mean * 100}%`,
            borderRadius: '50%',
            background: tone,
            border: '2px solid var(--color-surface)',
            transition: 'left 0.6s ease',
          }}
        />
      </div>
    </div>
  );
}
