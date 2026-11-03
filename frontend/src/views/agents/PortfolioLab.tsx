// ═══════════════════════════════════════════════════════
// FinSight web — Portfolio allocator lab
// ═══════════════════════════════════════════════════════
// Configure → train → watch the curve → read an out-of-sample backtest →
// hand the target allocation to the paper portfolio as proposals.
// ═══════════════════════════════════════════════════════

import { useState } from 'react';
import * as api from '../../api/client';
import type { PortfolioResult, StrategyMetrics } from '../../api/types';
import { Card, Icon, Slider, TickerInput, useAction } from '../../components/ui';
import type { View } from '../../hooks/useRoute';
import { pct, signedPct } from '../../lib/money';
import { Donut, Legend, SERIES, TrendChart } from '../../viz/finance';
import { LivePanel, RunHistory, useLiveRun } from './shared';

export const PRESETS: Record<string, string[]> = {
  'Big tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
  'Index ETFs': ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
  Defensive: ['JNJ', 'PG', 'KO', 'XLU', 'TLT'],
};

export const SELECTED_RUN_KEY = 'finsight.selectedRun';

export function riskLabel(r: number): string {
  return r < 0.75 ? 'Aggressive' : r < 2.5 ? 'Balanced' : 'Conservative';
}

export interface MarketParams {
  tickers: string[];
  risk_aversion: number;
  episodes: number;
  cost_bps: number;
}

/** The ticker / risk / episodes / cost form both market agents share. */
export function MarketForm({
  title,
  value,
  onChange,
  onTrain,
  disabled,
  training,
  riskHint,
}: {
  title: string;
  value: MarketParams;
  onChange: (v: MarketParams) => void;
  onTrain: () => void;
  disabled: boolean;
  training: boolean;
  riskHint: string;
}) {
  const set = (patch: Partial<MarketParams>) => onChange({ ...value, ...patch });
  return (
    <Card title={title}>
      <div style={{ display: 'grid', gap: 18 }}>
        <div className="field">
          <label>Universe (2–8 tickers)</label>
          <TickerInput value={value.tickers} onChange={(tickers) => set({ tickers })} />
          <div className="row wrap-row" style={{ gap: 6, marginTop: 8 }}>
            {Object.entries(PRESETS).map(([name, list]) => (
              <button key={name} type="button" className="chip" style={{ border: 0, cursor: 'pointer' }} onClick={() => set({ tickers: list })}>
                {name}
              </button>
            ))}
          </div>
        </div>
        <Slider
          label="Risk aversion"
          value={value.risk_aversion}
          min={0}
          max={5}
          step={0.25}
          onChange={(risk_aversion) => set({ risk_aversion })}
          format={(n) => `${riskLabel(n)} · ${n.toFixed(2)}`}
          hint={riskHint}
        />
        <Slider
          label="Training episodes"
          value={value.episodes}
          min={100}
          max={3000}
          step={100}
          onChange={(episodes) => set({ episodes })}
          format={(n) => n.toLocaleString()}
          hint="Each episode replays 60 trading days. More episodes, steadier policy."
        />
        <Slider
          label="Trading cost"
          value={value.cost_bps}
          min={0}
          max={50}
          step={1}
          onChange={(cost_bps) => set({ cost_bps })}
          format={(n) => `${n} bps`}
          hint="Charged on every change of position, so the agent learns not to churn."
        />
        <button className="btn btn-primary" type="button" onClick={onTrain} disabled={disabled || training || value.tickers.length < 2}>
          <Icon name="play" size={16} /> {training ? 'Training…' : 'Train agent'}
        </button>
      </div>
    </Card>
  );
}

export function TickerChips({ tickers, params }: { tickers: string[]; params: { risk_aversion?: number; cost_bps?: number } }) {
  return (
    <>
      {tickers.map((t, i) => (
        <span key={t} className="chip" style={{ background: `color-mix(in srgb, ${SERIES[i % SERIES.length]} 14%, transparent)`, color: SERIES[i % SERIES.length] }}>
          {t}
        </span>
      ))}
      <span className="chip">risk {params.risk_aversion ?? '—'}</span>
      <span className="chip">{params.cost_bps ?? '—'} bps</span>
    </>
  );
}

export function PortfolioLab({ navigate }: { navigate: (v: View) => void }) {
  const live = useLiveRun<PortfolioResult>('portfolio');
  const [params, setParams] = useState<MarketParams>({ tickers: PRESETS['Index ETFs'], risk_aversion: 1, episodes: 800, cost_bps: 10 });
  const { busy, run } = useAction();
  const active = live.active;

  return (
    <>
      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <MarketForm
          title="Train the portfolio allocator"
          value={params}
          onChange={setParams}
          disabled={busy}
          training={live.training}
          riskHint="Higher values penalise volatile days more, trading return for a smoother ride."
          onTrain={async () => {
            const r = await run(() => api.startRun(params), 'Training started — watch the curve fill in');
            if (r) live.started(r);
          }}
        />
        <LivePanel
          run={active}
          emptyIcon="brain"
          empty={<>Pick a universe and press <b>Train agent</b>. The learning curve draws itself here as the agent improves.</>}
          curveLabel="Mean episode reward vs an equal-weight basket (higher is better)"
          chips={active ? <TickerChips tickers={active.tickers} params={active.params} /> : null}
        />
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

      <RunHistory
        runs={live.runs.data}
        activeId={live.activeId}
        onSelect={live.setActiveId}
        columns={[
          { label: 'Universe', render: (r) => r.tickers.join(', ') },
          { label: 'Return', right: true, render: (r) => (r.result?.metrics ? signedPct(r.result.metrics.policy.total_return) : '—') },
          { label: 'Sharpe', right: true, render: (r) => (r.result?.metrics ? r.result.metrics.policy.sharpe.toFixed(2) : '—') },
          {
            label: 'vs equal-weight',
            right: true,
            render: (r) => {
              const m = r.result?.metrics;
              if (!m) return '—';
              const d = m.policy.total_return - m.equal_weight.total_return;
              return <span className={d >= 0 ? 'pos' : 'neg'}>{signedPct(d)}</span>;
            },
          },
        ]}
      />
    </>
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
