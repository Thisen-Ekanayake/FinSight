// ═══════════════════════════════════════════════════════
// FinSight web — Overview
// ═══════════════════════════════════════════════════════
// The home screen: this month's money, goals, the latest agent, the paper
// account, and anything waiting on a decision — each a click from its view.
// ═══════════════════════════════════════════════════════

import * as api from '../api/client';
import { Card, Disclaimer, Icon, ProgressBar, Ring, Skeleton, Stat } from '../components/ui';
import type { View } from '../hooks/useRoute';
import { useResource } from '../hooks/useResource';
import { money, monthLabel, pct, signedPct, thisMonth, usageTone } from '../lib/money';
import { Donut, SERIES } from '../viz/finance';
import { StatusChip } from './Agents';

function greeting(): string {
  const h = new Date().getHours();
  return h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
}

export function Overview({ navigate, pendingCount, email }: { navigate: (v: View) => void; pendingCount: number; email?: string }) {
  const month = thisMonth();
  const summary = useResource(() => api.financeSummary(month), []);
  const runs = useResource(() => api.listPolicyRuns(5, 'portfolio'), []);
  const broker = useResource(() => api.brokerStatus(), []);
  const proposals = useResource(() => api.listProposals('PENDING'), []);

  const s = summary.data;
  const latest = runs.data?.find((r) => r.status === 'DONE') ?? runs.data?.[0];
  const equity = broker.data?.account?.equity ?? 0;
  const name = email && email !== 'anonymous' ? email.split('@')[0] : '';
  const topCats = [...(s?.categories ?? [])].sort((a, b) => b.spent / (b.monthly_limit || 1) - a.spent / (a.monthly_limit || 1)).slice(0, 5);

  return (
    <div className="page">
      <div className="hero rise" style={{ marginBottom: 18 }}>
        <div className="row between wrap-row" style={{ gap: 18, position: 'relative', zIndex: 1 }}>
          <div>
            <div style={{ opacity: 0.85, fontSize: 14 }}>{greeting()}{name ? `, ${name}` : ''}</div>
            <h1 style={{ color: '#fff', margin: '6px 0 4px', fontSize: 34 }}>
              {s ? `${s.net >= 0 ? '+' : '−'}${money(Math.abs(s.net), true)}` : '—'}
            </h1>
            <div style={{ opacity: 0.85 }}>net for {monthLabel(month)} · saving {pct(s?.savings_rate ?? 0, 0)} of income</div>
          </div>
          <div className="row wrap-row" style={{ gap: 8 }}>
            <button className="btn" type="button" onClick={() => navigate('budget')} style={{ background: 'rgba(255,255,255,0.16)', color: '#fff', borderColor: 'rgba(255,255,255,0.25)' }}>
              <Icon name="plus" size={16} /> Add transaction
            </button>
            <button className="btn" type="button" onClick={() => navigate('agents')} style={{ background: '#fff', color: 'var(--color-accent-700)', borderColor: 'transparent' }}>
              <Icon name="brain" size={16} /> Train an agent
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 18 }}>
        <Stat label="Income this month" value={s?.income ?? 0} format={(n) => money(n, true)} icon="down" tone="var(--pos)" />
        <Stat
          label="Spent this month"
          value={s?.expenses ?? 0}
          format={(n) => money(n, true)}
          icon="up"
          tone="var(--neg)"
          footer={s && s.budget_total ? <ProgressBar value={s.expenses} max={s.budget_total} tone={usageTone(s.expenses, s.budget_total)} height={5} /> : 'No budget yet'}
        />
        <Stat
          label="Saved toward goals"
          value={s?.goals_saved ?? 0}
          format={(n) => money(n, true)}
          icon="target"
          tone="var(--series-3)"
          footer={s && s.goals_target ? `${pct(s.goals_saved / s.goals_target, 0)} of ${money(s.goals_target, true)}` : 'No goals yet'}
        />
        <Stat
          label="Paper portfolio"
          value={equity}
          format={(n) => (broker.data?.linked ? money(n, true) : 'Not linked')}
          icon="chart"
          tone="var(--series-2)"
          footer={broker.data?.linked ? `${broker.data.positions.length} positions` : <a href="/portfolio" onClick={(e) => { e.preventDefault(); navigate('portfolio'); }}>Link Alpaca paper →</a>}
        />
      </div>

      {pendingCount > 0 || (proposals.data?.length ?? 0) > 0 ? (
        <div className="card rise row wrap-row" style={{ padding: 14, marginBottom: 18, gap: 12, borderColor: 'var(--warn)' }}>
          <Icon name="bolt" style={{ color: 'var(--warn)' }} />
          <span style={{ flex: 1 }}>
            <b>Waiting on you:</b>{' '}
            {[proposals.data?.length ? `${proposals.data.length} order proposal(s)` : '', pendingCount ? `${pendingCount} research alert(s)` : '']
              .filter(Boolean)
              .join(' and ')}
          </span>
          {proposals.data?.length ? <button className="btn btn-sm" type="button" onClick={() => navigate('portfolio')}>Review orders</button> : null}
          {pendingCount ? <button className="btn btn-sm" type="button" onClick={() => navigate('desk')}>Review alerts</button> : null}
        </div>
      ) : null}

      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <Card title="Budget health" action={<button className="btn btn-ghost btn-sm" type="button" onClick={() => navigate('budget')}>Open budget →</button>}>
          {!s ? (
            <Skeleton height={150} />
          ) : topCats.length ? (
            <div style={{ display: 'grid', gap: 14 }}>
              {topCats.map((c) => (
                <ProgressBar
                  key={c.name}
                  value={c.spent}
                  max={c.monthly_limit || Math.max(c.spent, 1)}
                  tone={usageTone(c.spent, c.monthly_limit)}
                  label={c.name}
                  right={`${money(c.spent, true)} / ${c.monthly_limit ? money(c.monthly_limit, true) : '—'}`}
                />
              ))}
            </div>
          ) : (
            <p className="text-muted" style={{ margin: 0 }}>Set up categories to see how each one is tracking.</p>
          )}
        </Card>

        <Card title="Savings goals" action={<button className="btn btn-ghost btn-sm" type="button" onClick={() => navigate('budget')}>Manage →</button>}>
          {s?.goals.length ? (
            <div style={{ display: 'grid', gap: 12 }}>
              {s.goals.slice(0, 4).map((g) => {
                const f = g.target ? g.saved / g.target : 0;
                return (
                  <div key={g.id} className="row" style={{ gap: 12 }}>
                    <Ring value={f} size={46} stroke={6} tone={f >= 1 ? 'var(--pos)' : 'var(--color-accent)'}>
                      <span className="num" style={{ fontSize: 10.5, fontWeight: 700 }}>{pct(Math.min(f, 9.99), 0)}</span>
                    </Ring>
                    <div style={{ minWidth: 0 }}>
                      <b>{g.name}</b>
                      <div className="num" style={{ fontSize: 12.5, color: 'var(--ink-58)' }}>{money(g.saved, true)} of {money(g.target, true)}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-muted" style={{ margin: 0 }}>No goals yet.</p>
          )}
        </Card>

        <Card title="Portfolio agent" action={latest ? <StatusChip status={latest.status} /> : null}>
          {latest?.result?.target_weights ? (
            <>
              <Donut
                size={130}
                thickness={20}
                data={Object.entries(latest.result.target_weights).map(([label, value], i) => ({ label, value, color: label === 'CASH' ? 'var(--ink-42)' : SERIES[i % SERIES.length] }))}
                center={<span style={{ fontSize: 11, color: 'var(--ink-52)' }}>target</span>}
              />
              {latest.result.metrics ? (
                <div className="row between" style={{ marginTop: 12, fontSize: 13 }}>
                  <span className="text-muted">Backtest vs equal-weight</span>
                  <b className={latest.result.metrics.policy.total_return >= latest.result.metrics.equal_weight.total_return ? 'pos' : 'neg'}>
                    {signedPct(latest.result.metrics.policy.total_return - latest.result.metrics.equal_weight.total_return)}
                  </b>
                </div>
              ) : null}
            </>
          ) : latest && (latest.status === 'RUNNING' || latest.status === 'QUEUED') ? (
            <ProgressBar value={latest.progress} striped label={latest.stage} right={pct(latest.progress, 0)} />
          ) : (
            <div style={{ textAlign: 'center', padding: '8px 0' }}>
              <p className="text-muted">Train the allocator on a few tickers to get a target portfolio.</p>
              <button className="btn btn-primary btn-sm" type="button" onClick={() => navigate('agents')}>
                <Icon name="play" size={14} /> Start training
              </button>
            </div>
          )}
        </Card>
      </div>

      <Disclaimer />
    </div>
  );
}
