// ═══════════════════════════════════════════════════════
// FinSight web — Trade timing lab
// ═══════════════════════════════════════════════════════
// Long / flat signals per ticker, each judged out of sample against simply
// holding it. Signals only: this agent has no path to the broker.
// ═══════════════════════════════════════════════════════

import { useEffect, useState } from 'react';
import * as api from '../../api/client';
import type { TimingResult, TimingSignal } from '../../api/types';
import { Card, Disclaimer, Icon, ProgressBar, useAction } from '../../components/ui';
import { pct, signedPct } from '../../lib/money';
import { Legend, TrendChart } from '../../viz/finance';
import { MarketForm, PRESETS, TickerChips, type MarketParams } from './PortfolioLab';
import { LivePanel, RunHistory, useLiveRun } from './shared';

const FEATURE_LABELS: Record<string, string> = {
  mom_5d: '1-week momentum',
  mom_20d: '1-month momentum',
  mom_60d: '3-month momentum',
  vol_ratio: 'Volatility rising',
  vs_ma50: 'Above 50-day average',
  vol_level: 'Volatility level',
  position: 'Already invested',
  bias: 'Default lean to “long”',
};

export function TimingLab() {
  const live = useLiveRun<TimingResult>('timing');
  const [params, setParams] = useState<MarketParams>({ tickers: PRESETS['Big tech'], risk_aversion: 1, episodes: 600, cost_bps: 10 });
  const { busy, run } = useAction();
  const active = live.active;
  const result = active?.status === 'DONE' ? active.result : null;

  return (
    <>
      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <MarketForm
          title="Train the timing agent"
          value={params}
          onChange={setParams}
          disabled={busy}
          training={live.training}
          riskHint="Higher values make volatile stretches costlier to sit through, so the agent steps aside sooner."
          onTrain={async () => {
            const r = await run(() => api.startTimingRun(params), 'Timing agent is training');
            if (r) live.started(r);
          }}
        />
        <LivePanel
          run={active}
          emptyIcon="bolt"
          empty={
            <>
              The agent decides once a week whether each ticker is worth holding or sitting out. It only scores points for getting
              out before falls — holding through everything scores zero.
            </>
          }
          curveLabel="Mean episode reward vs buy-and-hold (above 0 = beat holding)"
          chips={active ? <TickerChips tickers={active.tickers} params={active.params} /> : null}
        />
      </div>

      {result ? <TimingResults result={result} /> : null}

      <RunHistory
        runs={live.runs.data}
        activeId={live.activeId}
        onSelect={live.setActiveId}
        columns={[
          { label: 'Universe', render: (r) => r.tickers.join(', ') },
          {
            label: 'Signals',
            render: (r) =>
              r.result?.signals ? (
                <span className="row wrap-row" style={{ gap: 4 }}>
                  {r.result.signals.map((s) => (
                    <span key={s.ticker} className={`chip ${s.signal === 'LONG' ? 'chip-pos' : 'chip-warn'}`} style={{ fontSize: 10.5, padding: '1px 7px' }}>
                      {s.ticker} {s.signal}
                    </span>
                  ))}
                </span>
              ) : (
                '—'
              ),
          },
        ]}
      />
    </>
  );
}

function SignalCard({ s, selected, onSelect }: { s: TimingSignal; selected: boolean; onSelect: () => void }) {
  const long = s.signal === 'LONG';
  const dSharpe = s.metrics.agent.sharpe - s.metrics.buy_hold.sharpe;
  return (
    <button
      type="button"
      onClick={onSelect}
      className="card card-hover"
      style={{
        padding: 16,
        textAlign: 'left',
        font: 'inherit',
        color: 'inherit',
        cursor: 'pointer',
        borderColor: selected ? 'var(--color-accent)' : undefined,
        boxShadow: selected ? '0 0 0 3px color-mix(in srgb, var(--color-accent) 20%, transparent)' : undefined,
      }}
    >
      <div className="row between" style={{ marginBottom: 10 }}>
        <b style={{ fontSize: 17 }}>{s.ticker}</b>
        <span className={`chip ${long ? 'chip-pos' : 'chip-warn'}`}>
          <Icon name={long ? 'up' : 'pause'} size={12} /> {s.signal}
        </span>
      </div>
      <div style={{ display: 'grid', gap: 10 }}>
        <ProgressBar value={s.confidence} tone={long ? 'var(--pos)' : 'var(--warn)'} height={6} label={<span style={{ fontSize: 12 }}>Confidence</span>} right={pct(s.confidence, 0)} />
        <ProgressBar value={s.time_in_market} height={6} label={<span style={{ fontSize: 12 }}>Time invested (test)</span>} right={pct(s.time_in_market, 0)} />
      </div>
      <div className="row between" style={{ marginTop: 10, fontSize: 12.5 }}>
        <span className="text-muted">Sharpe vs holding</span>
        <b className={`num ${dSharpe >= 0 ? 'pos' : 'neg'}`}>
          {s.metrics.agent.sharpe.toFixed(2)} / {s.metrics.buy_hold.sharpe.toFixed(2)}
        </b>
      </div>
    </button>
  );
}

function TimingResults({ result }: { result: TimingResult }) {
  const [ticker, setTicker] = useState(result.signals[0]?.ticker);
  useEffect(() => setTicker(result.signals[0]?.ticker), [result]);
  const sel = result.signals.find((s) => s.ticker === ticker) ?? result.signals[0];
  const weights = Object.entries(result.feature_weights);
  const maxW = Math.max(...weights.map(([, w]) => Math.abs(w)), 0.01);
  const beat = result.signals.filter((s) => s.metrics.agent.sharpe > s.metrics.buy_hold.sharpe).length;

  return (
    <>
      <div className="row between wrap-row" style={{ marginBottom: 12, gap: 10 }}>
        <h3 style={{ margin: 0 }}>Today's signals</h3>
        <span className="chip">
          Beat holding (Sharpe) on {beat} of {result.signals.length} · tested {result.test_span.start} → {result.test_span.end}
        </span>
      </div>
      <div className="grid grid-4" style={{ marginBottom: 18 }}>
        {result.signals.map((s) => (
          <SignalCard key={s.ticker} s={s} selected={s.ticker === sel?.ticker} onSelect={() => setTicker(s.ticker)} />
        ))}
      </div>

      {sel ? (
        <div className="grid grid-3" style={{ marginBottom: 18 }}>
          <Card
            className="span-2"
            title={
              <div>
                <h3 style={{ margin: 0 }}>{sel.ticker}: agent vs holding, out of sample</h3>
                <div style={{ fontSize: 12.5, color: 'var(--ink-52)' }}>
                  {sel.switches} switches · total {signedPct(sel.metrics.agent.total_return)} vs {signedPct(sel.metrics.buy_hold.total_return)} ·
                  max drawdown −{pct(sel.metrics.agent.max_drawdown)} vs −{pct(sel.metrics.buy_hold.max_drawdown)}
                </div>
              </div>
            }
          >
            <TrendChart
              labels={result.dates}
              series={[
                { name: 'Timing agent', values: sel.equity.agent, color: 'var(--series-1)', area: true },
                { name: 'Buy & hold', values: sel.equity.buy_hold, color: 'var(--series-3)', dash: '6 4' },
              ]}
              format={(n) => `${((n - 1) * 100).toFixed(1)}%`}
              height={240}
            />
            {/* In-market strip: one cell per (downsampled) bar. */}
            <div style={{ display: 'flex', height: 10, borderRadius: 6, overflow: 'hidden', margin: '8px 12px 0 52px' }} aria-label="Invested (green) or in cash (amber) over the test period">
              {sel.equity.position.map((p, i) => (
                <span key={i} style={{ flex: 1, background: p ? 'var(--pos)' : 'var(--warn)', opacity: p ? 0.75 : 0.9 }} />
              ))}
            </div>
            <Legend
              series={[
                { name: 'Timing agent', color: 'var(--series-1)' },
                { name: 'Buy & hold', color: 'var(--series-3)', dash: '6 4' },
                { name: 'Strip: invested / in cash', color: 'var(--pos)' },
              ]}
            />
          </Card>
          <Card title="What the policy looks at">
            <div style={{ display: 'grid', gap: 10 }}>
              {weights.map(([name, w]) => (
                <div key={name}>
                  <div className="row between" style={{ fontSize: 12.5, marginBottom: 4 }}>
                    <span>{FEATURE_LABELS[name] ?? name}</span>
                    <b className={`num ${w >= 0 ? 'pos' : 'neg'}`}>{w >= 0 ? '+' : '−'}{Math.abs(w).toFixed(2)}</b>
                  </div>
                  {/* Diverging bar: right of centre pushes toward LONG, left toward FLAT. */}
                  <div style={{ position: 'relative', height: 6, background: 'var(--color-field)', borderRadius: 6 }}>
                    <span style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 1, background: 'var(--ink-42)' }} />
                    <span
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        borderRadius: 6,
                        background: w >= 0 ? 'var(--pos)' : 'var(--neg)',
                        left: w >= 0 ? '50%' : `${50 - (Math.abs(w) / maxW) * 50}%`,
                        width: `${(Math.abs(w) / maxW) * 50}%`,
                        transition: 'all 0.6s ease',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <p style={{ fontSize: 12, color: 'var(--ink-52)', margin: '14px 0 0' }}>
              Green pushes toward staying invested, red toward cash.
            </p>
          </Card>
        </div>
      ) : null}

      <div style={{ marginBottom: 18 }}>
        <Disclaimer>
          Signals and backtests only — the timing agent cannot place orders. On real daily data it usually stays invested; timing the
          market rarely beats holding once costs are counted, and the comparison above is shown so you can judge for yourself.
        </Disclaimer>
      </div>
    </>
  );
}
