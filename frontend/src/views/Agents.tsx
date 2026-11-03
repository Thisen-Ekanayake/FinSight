// ═══════════════════════════════════════════════════════
// FinSight web — Agents (RL lab)
// ═══════════════════════════════════════════════════════
//
// Configure → train → watch the learning curve fill in live → read an
// out-of-sample backtest against two plain benchmarks → hand the target
// allocation to the paper portfolio as proposals.
//
// ══ WHY POLL AND NOT STREAM ══
//   The run's progress lives in the database (src/rl/service.py), so a
//   reload, a second tab or a phone all see the same bar. A 700 ms poll
//   while a run is live is cheap and needs no extra transport.
// ═══════════════════════════════════════════════════════

import { useEffect, useState } from 'react';
import * as api from '../api/client';
import type { PolicyRun, PortfolioResult, StrategyMetrics } from '../api/types';
import { Card, Disclaimer, Icon, PageTitle, ProgressBar, Slider, TickerInput, useAction } from '../components/ui';
import type { View } from '../hooks/useRoute';
import { useResource } from '../hooks/useResource';
import { pct, signedPct } from '../lib/money';
import { Donut, Legend, SERIES, TrendChart } from '../viz/finance';

const PRESETS: Record<string, string[]> = {
  'Big tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
  'Index ETFs': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
  Defensive: ['JNJ', 'PG', 'KO', 'XLU', 'TLT'],
};

export const SELECTED_RUN_KEY = 'finsight.selectedRun';

function riskLabel(r: number): string {
  return r < 0.75 ? 'Aggressive' : r < 2.5 ? 'Balanced' : 'Conservative';
}

export function Agents({ navigate }: { navigate: (v: View) => void }) {
  const agents = useResource(() => api.listAgents(), []);
  const runs = useResource(() => api.listPolicyRuns(15), []);

  const [tickers, setTickers] = useState<string[]>(PRESETS['Index ETFs']);
  const [risk, setRisk] = useState(1);
  const [episodes, setEpisodes] = useState(800);
  const [cost, setCost] = useState(10);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [active, setActive] = useState<PolicyRun | null>(null);
  const { busy, run } = useAction();

  // Default to the most recent run once the history arrives.
  useEffect(() => {
    if (!activeId && runs.data?.length) setActiveId(runs.data[0].run_id);
  }, [runs.data, activeId]);

  // Load the selected run, and keep polling it while it is live.
  useEffect(() => {
    if (!activeId) return;
    let live = true;
    let timer: number | undefined;
    const tick = async () => {
      try {
        const r = await api.getPolicyRun(activeId);
        if (!live) return;
        setActive(r);
        if (r.status === 'QUEUED' || r.status === 'RUNNING') timer = window.setTimeout(tick, 700);
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

  const training = active?.status === 'QUEUED' || active?.status === 'RUNNING';

  const start = async () => {
    const r = await run(
      () => api.startRun({ tickers, risk_aversion: risk, episodes, cost_bps: cost }),
      'Training started — watch the curve fill in',
    );
    if (r) {
      setActive(r);
      setActiveId(r.run_id);
      runs.reload();
    }
  };

  return (
    <div className="page">
      <PageTitle
        title="Agents"
        subtitle="Reinforcement-learning agents that learn from market history. Train one, judge it on data it never saw, then decide whether to act on it."
      />

      <div className="grid grid-4" style={{ marginBottom: 18 }}>
        {(agents.data ?? []).map((a) => (
          <div
            key={a.id}
            className="card card-hover rise"
            style={{
              padding: 16,
              opacity: a.status === 'available' ? 1 : 0.72,
              borderColor: a.status === 'available' ? 'color-mix(in srgb, var(--color-accent) 45%, transparent)' : undefined,
            }}
          >
            <div className="row between" style={{ marginBottom: 8 }}>
              <span style={{ display: 'grid', placeItems: 'center', width: 34, height: 34, borderRadius: 10, background: 'var(--color-accent-100)', color: 'var(--color-accent)' }}>
                <Icon name={a.id === 'portfolio' ? 'chart' : a.id === 'budget' ? 'wallet' : a.id === 'timing' ? 'bolt' : 'inbox'} size={18} />
              </span>
              <span className={`chip ${a.status === 'available' ? 'chip-pos' : ''}`}>{a.status === 'available' ? 'Live' : 'Planned'}</span>
            </div>
            <b>{a.name}</b>
            <p style={{ fontSize: 12.5, color: 'var(--ink-58)', margin: '4px 0 0' }}>{a.summary}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <Card title="Train the portfolio allocator">
          <div style={{ display: 'grid', gap: 18 }}>
            <div className="field">
              <label>Universe (2–8 tickers)</label>
              <TickerInput value={tickers} onChange={setTickers} />
              <div className="row wrap-row" style={{ gap: 6, marginTop: 8 }}>
                {Object.entries(PRESETS).map(([name, list]) => (
                  <button key={name} type="button" className="chip" style={{ border: 0, cursor: 'pointer' }} onClick={() => setTickers(list)}>
                    {name}
                  </button>
                ))}
              </div>
            </div>
            <Slider
              label="Risk aversion"
              value={risk}
              min={0}
              max={5}
              step={0.25}
              onChange={setRisk}
              format={(n) => `${riskLabel(n)} · ${n.toFixed(2)}`}
              hint="Higher values penalise volatile days more, trading return for a smoother ride."
            />
            <Slider
              label="Training episodes"
              value={episodes}
              min={100}
              max={3000}
              step={100}
              onChange={setEpisodes}
              format={(n) => n.toLocaleString()}
              hint="Each episode replays 60 trading days. More episodes, steadier policy."
            />
            <Slider label="Trading cost" value={cost} min={0} max={50} step={1} onChange={setCost} format={(n) => `${n} bps`} hint="Charged on every rebalance, so the agent learns not to churn." />
            <button className="btn btn-primary" type="button" onClick={start} disabled={busy || training || tickers.length < 2}>
              <Icon name="play" size={16} /> {training ? 'Training…' : 'Train agent'}
            </button>
          </div>
        </Card>

        <Card className="span-2" title="Live training" action={active ? <StatusChip status={active.status} /> : null}>
          {!active ? (
            <div style={{ display: 'grid', placeItems: 'center', height: 320, textAlign: 'center', color: 'var(--ink-52)' }}>
              <div>
                <Icon name="brain" size={42} style={{ color: 'var(--color-accent)' }} />
                <p style={{ marginTop: 12 }}>Pick a universe and press <b>Train agent</b>. The learning curve draws itself here as the agent improves.</p>
              </div>
            </div>
          ) : (
            <>
              <div className="row wrap-row" style={{ gap: 8, marginBottom: 12 }}>
                {active.tickers.map((t, i) => (
                  <span key={t} className="chip" style={{ background: `color-mix(in srgb, ${SERIES[i % SERIES.length]} 14%, transparent)`, color: SERIES[i % SERIES.length] }}>
                    {t}
                  </span>
                ))}
                <span className="chip">risk {active.params.risk_aversion ?? '—'}</span>
                <span className="chip">{active.params.cost_bps ?? '—'} bps</span>
              </div>
              <ProgressBar
                value={active.progress}
                striped={training}
                height={12}
                tone={active.status === 'FAILED' ? 'var(--neg)' : active.status === 'DONE' ? 'var(--pos)' : undefined}
                label={<span style={{ fontWeight: 500 }}>{active.stage || active.status}</span>}
                right={`${active.episodes_done.toLocaleString()} / ${active.episodes_total.toLocaleString()} episodes · ${pct(active.progress, 0)}`}
              />
              {active.error ? <p className="chip chip-neg" style={{ marginTop: 12, whiteSpace: 'normal' }}>{active.error}</p> : null}
              <div style={{ marginTop: 18 }}>
                <div className="kicker" style={{ marginBottom: 6 }}>Mean episode reward (higher is better)</div>
                <TrendChart
                  labels={active.curve.map((_, i) => `Batch ${i + 1}`)}
                  series={[{ name: 'Reward', values: active.curve, color: 'var(--series-1)', area: true }]}
                  format={(n) => n.toFixed(3)}
                  height={220}
                  baseline={0}
                />
              </div>
            </>
          )}
        </Card>
      </div>

      {active?.status === 'DONE' && active.result ? (
        <Results
          result={active.result}
          onSend={() => {
            try {
              sessionStorage.setItem(SELECTED_RUN_KEY, active.run_id);
            } catch {
              /* storage blocked: Portfolio falls back to the latest run */
            }
            navigate('portfolio');
          }}
        />
      ) : null}

      <Card title="Run history" style={{ marginBottom: 18 }}>
        {runs.data?.length ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Started</th>
                  <th>Universe</th>
                  <th>Status</th>
                  <th className="r">Return</th>
                  <th className="r">Sharpe</th>
                  <th className="r">vs equal-weight</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.map((r) => {
                  const m = r.result?.metrics;
                  return (
                    <tr key={r.run_id} onClick={() => setActiveId(r.run_id)} style={{ cursor: 'pointer', background: r.run_id === activeId ? 'var(--color-accent-100)' : undefined }}>
                      <td className="num">{r.created_at ? new Date(r.created_at).toLocaleString() : '—'}</td>
                      <td>{r.tickers.join(', ')}</td>
                      <td><StatusChip status={r.status} /></td>
                      <td className="r num">{m ? signedPct(m.policy.total_return) : '—'}</td>
                      <td className="r num">{m ? m.policy.sharpe.toFixed(2) : '—'}</td>
                      <td className="r num">
                        {m ? <span className={m.policy.total_return >= m.equal_weight.total_return ? 'pos' : 'neg'}>{signedPct(m.policy.total_return - m.equal_weight.total_return)}</span> : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-muted" style={{ margin: 0 }}>No runs yet.</p>
        )}
      </Card>

      <Disclaimer />
    </div>
  );
}

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
      {live ? <span style={{ width: 7, height: 7, borderRadius: 9, background: 'currentColor', animation: 'fs-pulse 1s infinite' }} /> : null}
      {status.charAt(0) + status.slice(1).toLowerCase()}
    </span>
  );
}

const STRATEGIES: { key: keyof PortfolioResult['metrics']; name: string; color: string; dash?: string }[] = [
  { key: 'policy', name: 'RL agent', color: 'var(--series-1)' },
  { key: 'equal_weight', name: 'Equal weight', color: 'var(--series-2)', dash: '6 4' },
  { key: 'buy_hold', name: 'Buy & hold', color: 'var(--series-3)', dash: '2 4' },
];

const METRIC_ROWS: { key: keyof StrategyMetrics; label: string; fmt: (n: number) => string; better: 'high' | 'low' }[] = [
  { key: 'total_return', label: 'Total return', fmt: (n) => signedPct(n), better: 'high' },
  { key: 'annual_return', label: 'Annualised return', fmt: (n) => signedPct(n), better: 'high' },
  { key: 'annual_volatility', label: 'Volatility', fmt: (n) => pct(n), better: 'low' },
  { key: 'sharpe', label: 'Sharpe ratio', fmt: (n) => n.toFixed(2), better: 'high' },
  { key: 'max_drawdown', label: 'Max drawdown', fmt: (n) => `−${pct(n)}`, better: 'low' },
];

function Results({ result, onSend }: { result: PortfolioResult; onSend: () => void }) {
  const weights = Object.entries(result.target_weights);
  return (
    <div className="grid grid-3" style={{ marginBottom: 18 }}>
      <Card
        className="span-2"
        title={
          <div>
            <h3 style={{ margin: 0 }}>Out-of-sample backtest</h3>
            <div style={{ fontSize: 12.5, color: 'var(--ink-52)' }}>
              Trained on {result.train_span.start} → {result.train_span.end}; tested on {result.test_span.start} → {result.test_span.end} ({result.test_span.bars} days it never saw)
            </div>
          </div>
        }
      >
        <TrendChart
          labels={result.backtest.dates}
          series={STRATEGIES.map((s) => ({ name: s.name, values: result.backtest[s.key], color: s.color, dash: s.dash, area: s.key === 'policy' }))}
          format={(n) => `${((n - 1) * 100).toFixed(1)}%`}
          baseline={1}
          height={260}
        />
        <Legend series={STRATEGIES} />
        <div style={{ overflowX: 'auto', marginTop: 16 }}>
          <table className="table">
            <thead>
              <tr>
                <th>Metric</th>
                {STRATEGIES.map((s) => (
                  <th key={s.key} className="r">{s.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map((row) => {
                const vals = STRATEGIES.map((s) => result.metrics[s.key][row.key]);
                const best = row.better === 'high' ? Math.max(...vals) : Math.min(...vals);
                return (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    {vals.map((v, i) => (
                      <td key={i} className="r num" style={{ fontWeight: v === best ? 700 : 400 }}>
                        {row.fmt(v)} {v === best ? <span className="chip chip-pos" style={{ padding: '0 6px', fontSize: 10 }}>best</span> : null}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
      <Card title="Target allocation today">
        <Donut
          size={170}
          data={weights.map(([label, value], i) => ({ label, value, color: label === 'CASH' ? 'var(--ink-42)' : SERIES[i % SERIES.length] }))}
          center={
            <div>
              <div style={{ fontSize: 11.5, color: 'var(--ink-52)' }}>Invested</div>
              <b className="num" style={{ fontFamily: 'var(--font-heading)', fontSize: 22 }}>{pct(1 - (result.target_weights.CASH ?? 0), 0)}</b>
            </div>
          }
        />
        <p style={{ fontSize: 12.5, color: 'var(--ink-58)', marginTop: 14 }}>
          What the agent would hold given the most recent {result.config.window} trading days. Orders cap any one position at
          your account&apos;s limit (40% by default), and still need your approval.
        </p>
        <button className="btn btn-primary" type="button" onClick={onSend} style={{ width: '100%' }}>
          <Icon name="right" size={16} /> Propose paper orders
        </button>
      </Card>
    </div>
  );
}
