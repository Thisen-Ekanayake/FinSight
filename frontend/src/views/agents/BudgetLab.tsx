// ═══════════════════════════════════════════════════════
// FinSight web — Budget & savings planner lab
// ═══════════════════════════════════════════════════════
// Train the bandit on your own ledger, see how every savings rate scores,
// drag a what-if slider, and apply the plan's limits with one click.
// ═══════════════════════════════════════════════════════

import { useEffect, useState } from 'react';
import * as api from '../../api/client';
import type { BudgetResult } from '../../api/types';
import { Card, Delta, Icon, ProgressBar, Ring, Slider, useAction } from '../../components/ui';
import { useResource } from '../../hooks/useResource';
import { money, pct } from '../../lib/money';
import { Legend, TrendChart } from '../../viz/finance';
import { LivePanel, RunHistory, useLiveRun } from './shared';

export function BudgetLab() {
  const live = useLiveRun<BudgetResult>('budget');
  const [rounds, setRounds] = useState(3000);
  const { busy, run } = useAction();
  const active = live.active;
  const result = active?.status === 'DONE' ? active.result : null;

  return (
    <>
      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <Card title="Plan my budget">
          <div style={{ display: 'grid', gap: 16 }}>
            <p style={{ margin: 0, color: 'var(--ink-62)', fontSize: 14 }}>
              The planner reads your income, fixed needs, flexible wants and savings goals, builds a simulator of
              <i> your</i> household, and lets a contextual bandit practise thousands of months in it to learn how much to
              set aside first.
            </p>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: 'var(--ink-62)', lineHeight: 1.8 }}>
              <li>Saves more when goals are behind or the emergency fund is thin</li>
              <li>Saves less when needs leave little room</li>
              <li>Never moves money — it only proposes limits</li>
            </ul>
            <Slider
              label="Simulated months"
              value={rounds}
              min={500}
              max={10000}
              step={500}
              onChange={setRounds}
              format={(n) => n.toLocaleString()}
              hint="More practice, a more confident recommendation."
            />
            <button
              className="btn btn-primary"
              type="button"
              disabled={busy || live.training}
              onClick={async () => {
                const r = await run(() => api.startBudgetRun(rounds), 'Planner is practising on your household');
                if (r) live.started(r);
              }}
            >
              <Icon name="play" size={16} /> {live.training ? 'Training…' : 'Train planner'}
            </button>
          </div>
        </Card>
        <LivePanel
          run={active}
          emptyIcon="wallet"
          empty={<>Add a few months of income and spending on the Budget page, then press <b>Train planner</b>.</>}
          curveLabel="Average monthly reward in simulation (higher is better)"
          unit="months"
          extraSeries={
            result ? [{ name: 'Always save 20%', values: result.rule_curve, color: 'var(--series-3)', dash: '6 4' }] : undefined
          }
        />
      </div>

      {result && active ? <BudgetResults runId={active.run_id} result={result} /> : null}

      <RunHistory
        runs={live.runs.data}
        activeId={live.activeId}
        onSelect={live.setActiveId}
        columns={[
          { label: 'Months simulated', right: true, render: (r) => (r.params.rounds ?? r.episodes_total).toLocaleString() },
          {
            label: 'Recommended rate',
            right: true,
            render: (r) => (r.result?.recommended_rate !== undefined && r.result?.recommended_rate !== null ? pct(r.result.recommended_rate, 0) : '—'),
          },
        ]}
      />
    </>
  );
}

function BudgetResults({ runId, result }: { runId: string; result: BudgetResult }) {
  const p = result.profile;
  const plan = result.plan;
  const [whatIf, setWhatIf] = useState(result.recommended_rate);
  useEffect(() => setWhatIf(result.recommended_rate), [result.recommended_rate]);
  const categories = useResource(() => api.listCategories(), [runId]);
  const { busy, run } = useAction();

  const idx = result.rates.indexOf(whatIf);
  const best = Math.max(...result.simulated_value);
  const wiSavings = whatIf * p.income;
  const wiLeft = p.income - p.needs - wiSavings;
  const current = Object.fromEntries((categories.data ?? []).map((c) => [c.name, c.monthly_limit]));

  return (
    <>
      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <Card title="Recommendation">
          <div className="row" style={{ gap: 18, alignItems: 'center' }}>
            <Ring value={result.recommended_rate / 0.5} size={110} stroke={11} tone="var(--pos)">
              <div>
                <b className="num" style={{ fontFamily: 'var(--font-heading)', fontSize: 26 }}>{pct(result.recommended_rate, 0)}</b>
                <div style={{ fontSize: 11, color: 'var(--ink-52)' }}>of income</div>
              </div>
            </Ring>
            <div>
              <div className="num" style={{ fontFamily: 'var(--font-heading)', fontSize: 24, fontWeight: 600 }}>
                {money(plan.monthly_savings, true)}<span style={{ fontSize: 14, color: 'var(--ink-52)' }}>/month</span>
              </div>
              <div style={{ marginTop: 6 }}>
                <Delta value={result.value_vs_rule} format={(n) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(3)} vs the 20% rule`} />
              </div>
            </div>
          </div>
          <div style={{ display: 'grid', gap: 7, fontSize: 13, marginTop: 18 }}>
            <Fact label="Monthly income" value={money(p.income, true)} note={p.assumed_income ? 'assumed — no income recorded' : `${p.months_of_history} month(s) of history`} />
            <Fact label="Fixed needs" value={money(p.needs, true)} />
            <Fact label="Flexible wants" value={money(p.wants, true)} />
            <Fact label="Goals need (on schedule)" value={`${money(p.goal_required, true)}/mo`} />
            <Fact label="Emergency buffer" value={`${p.emergency_buffer_months.toFixed(1)} months of needs`} />
          </div>
        </Card>

        <Card
          className="span-2"
          title={
            <div>
              <h3 style={{ margin: 0 }}>How every savings rate scores for you</h3>
              <div style={{ fontSize: 12.5, color: 'var(--ink-52)' }}>
                What the agent learned, against the simulator's own Monte-Carlo average. Drag the slider to explore.
              </div>
            </div>
          }
        >
          <TrendChart
            labels={result.rates.map((r) => `Save ${pct(r, 0)}`)}
            series={[
              { name: 'Simulated value', values: result.simulated_value, color: 'var(--series-2)', area: true },
              { name: 'Agent’s estimate', values: result.learned_value, color: 'var(--series-1)', dash: '6 4' },
            ]}
            format={(n) => n.toFixed(2)}
            height={200}
          />
          <Legend
            series={[
              { name: 'Simulated value', color: 'var(--series-2)' },
              { name: 'Agent’s estimate', color: 'var(--series-1)', dash: '6 4' },
            ]}
          />
          <div className="grid grid-2" style={{ marginTop: 16, alignItems: 'center' }}>
            <Slider label="What if I saved…" value={whatIf} min={0} max={0.5} step={0.05} onChange={setWhatIf} format={(n) => pct(n, 0)} />
            <div style={{ display: 'grid', gap: 8 }}>
              <ProgressBar
                value={result.simulated_value[idx] ?? 0}
                max={best}
                tone={whatIf === result.recommended_rate ? 'var(--pos)' : 'var(--color-accent)'}
                label="Score vs the best rate"
                right={pct((result.simulated_value[idx] ?? 0) / best, 0)}
              />
              <div style={{ fontSize: 13 }}>
                Save <b className="num">{money(wiSavings, true)}</b>, leaving{' '}
                <b className={`num ${wiLeft < 0 ? 'neg' : ''}`}>{money(wiLeft, true)}</b> after needs
                {wiLeft < 0 ? ' — needs would not fit' : ''}.
              </div>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 18 }}>
        <Card
          title="Proposed category limits"
          action={
            <button
              className="btn btn-primary btn-sm"
              type="button"
              disabled={busy}
              onClick={async () => {
                await run(() => api.applyBudgetPlan(runId), 'Plan applied to your budget categories');
                categories.reload();
              }}
            >
              <Icon name="check" size={14} /> Apply limits
            </button>
          }
        >
          <div style={{ display: 'grid', gap: 12 }}>
            {Object.entries(plan.category_limits).map(([name, limit]) => {
              const before = current[name];
              const change = before === undefined ? null : limit - before;
              return (
                <ProgressBar
                  key={name}
                  value={limit}
                  max={Math.max(...Object.values(plan.category_limits), 1)}
                  label={<b style={{ fontWeight: 600 }}>{name}</b>}
                  right={
                    <span>
                      {money(limit, true)}
                      {change ? (
                        <span className={change > 0 ? 'pos' : 'neg'} style={{ marginLeft: 6, fontSize: 12 }}>
                          ({change > 0 ? '+' : '−'}{money(Math.abs(change), true)})
                        </span>
                      ) : null}
                    </span>
                  }
                />
              );
            })}
          </div>
          {plan.unassigned > 0 ? (
            <p style={{ fontSize: 12.5, color: 'var(--ink-58)', margin: '14px 0 0' }}>
              {money(plan.unassigned, true)}/month is left unassigned — wants are capped at 30% above your recent spending
              rather than inflated to fill the gap.
            </p>
          ) : null}
          <p style={{ fontSize: 12, color: 'var(--ink-52)', margin: '8px 0 0' }}>Applying changes limits only. No money moves.</p>
        </Card>

        <Card title="Savings plan by goal" action={<span className="chip chip-pos">{money(plan.monthly_savings, true)}/mo</span>}>
          {plan.goals.length ? (
            <div style={{ display: 'grid', gap: 14 }}>
              {plan.goals.map((g) => (
                <div key={g.goal_id}>
                  <ProgressBar
                    value={g.monthly}
                    max={plan.monthly_savings || 1}
                    tone={g.on_track ? 'var(--pos)' : 'var(--warn)'}
                    label={
                      <span className="row" style={{ gap: 8 }}>
                        <b style={{ fontWeight: 600 }}>{g.name}</b>
                        <span className={`chip ${g.on_track ? 'chip-pos' : 'chip-warn'}`} style={{ fontSize: 10.5, padding: '1px 7px' }}>
                          {g.on_track ? 'on track' : 'behind'}
                        </span>
                      </span>
                    }
                    right={`${money(g.monthly, true)}/mo`}
                  />
                  <div style={{ fontSize: 12, color: 'var(--ink-52)', marginTop: 4 }}>
                    {money(g.remaining, true)} to go ·{' '}
                    {g.months_to_complete !== null ? `done in ~${g.months_to_complete} months` : 'not funded this month'}
                    {g.deadline_months !== null ? ` · deadline in ${g.deadline_months} months` : ''}
                  </div>
                </div>
              ))}
              {plan.unallocated_savings > 0 ? (
                <ProgressBar value={plan.unallocated_savings} max={plan.monthly_savings || 1} tone="var(--series-2)" label="General savings / investing" right={`${money(plan.unallocated_savings, true)}/mo`} />
              ) : null}
            </div>
          ) : (
            <p className="text-muted" style={{ margin: 0 }}>
              No open goals — all {money(plan.monthly_savings, true)} goes to general savings or investing. Add a goal on the Budget page to
              have the planner fund it.
            </p>
          )}
        </Card>
      </div>
    </>
  );
}

function Fact({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="row between" style={{ gap: 8 }}>
      <span className="text-muted">{label}</span>
      <span style={{ textAlign: 'right' }}>
        <b className="num">{value}</b>
        {note ? <div style={{ fontSize: 11, color: 'var(--ink-45)' }}>{note}</div> : null}
      </span>
    </div>
  );
}
