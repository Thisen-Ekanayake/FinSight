// ═══════════════════════════════════════════════════════
// FinSight web — Budget view
// ═══════════════════════════════════════════════════════
// One month at a time: stats, a burn-down against even pace, per-category
// progress bars, a quick-add ledger, and savings goals with rings.
// ═══════════════════════════════════════════════════════

import { useState } from 'react';
import * as api from '../api/client';
import type { CategoryKind, MonthSummary, SavingsGoal } from '../api/types';
import { Card, Icon, PageTitle, ProgressBar, Ring, Skeleton, Stat, useAction } from '../components/ui';
import { useResource } from '../hooks/useResource';
import { money, monthLabel, pct, shiftMonth, thisMonth, today, usageTone } from '../lib/money';
import { Legend, TrendChart } from '../viz/finance';

const TEMPLATE: [string, number, CategoryKind][] = [
  ['Rent', 1200, 'need'],
  ['Groceries', 400, 'need'],
  ['Transport', 150, 'need'],
  ['Utilities', 180, 'need'],
  ['Dining out', 200, 'want'],
  ['Entertainment', 120, 'want'],
  ['Investing', 500, 'savings'],
];

const KIND_CHIP: Record<string, string> = { need: 'chip', want: 'chip chip-warn', savings: 'chip chip-pos' };

export function Budget() {
  const [month, setMonth] = useState(thisMonth());
  const summary = useResource(() => api.financeSummary(month), [month]);
  const entries = useResource(() => api.listEntries(month), [month]);
  const reload = () => {
    summary.reload();
    entries.reload();
  };
  const s = summary.data;

  return (
    <div className="page">
      <PageTitle
        title="Budget"
        subtitle="Where the month's money went, against the plan you set. Every bar fills as you spend."
        action={
          <div className="row" style={{ gap: 6 }}>
            <button className="icon-btn" type="button" aria-label="Previous month" onClick={() => setMonth(shiftMonth(month, -1))}>
              <Icon name="left" />
            </button>
            <span style={{ fontWeight: 600, minWidth: 140, textAlign: 'center' }}>{monthLabel(month)}</span>
            <button className="icon-btn" type="button" aria-label="Next month" onClick={() => setMonth(shiftMonth(month, 1))}>
              <Icon name="right" />
            </button>
          </div>
        }
      />

      {summary.error ? <p className="chip chip-neg">{summary.error}</p> : null}

      <div className="grid grid-4" style={{ marginBottom: 18 }}>
        <Stat label="Income" value={s?.income ?? 0} format={(n) => money(n, true)} icon="down" tone="var(--pos)" />
        <Stat label="Spent" value={s?.expenses ?? 0} format={(n) => money(n, true)} icon="up" tone="var(--neg)"
          footer={s && s.budget_total > 0 ? `${pct(s.expenses / s.budget_total, 0)} of ${money(s.budget_total, true)} planned` : 'No budget set yet'} />
        <Stat label="Net" value={s?.net ?? 0} format={(n) => money(n, true)} icon="wallet" tone="var(--color-accent)" />
        <div className="card card-hover rise row" style={{ padding: 18, gap: 16 }}>
          <Ring value={Math.max(0, s?.savings_rate ?? 0)} size={74} stroke={9} tone="var(--pos)">
            <b className="num" style={{ fontSize: 15 }}>{pct(s?.savings_rate ?? 0, 0)}</b>
          </Ring>
          <div>
            <div style={{ fontSize: 13, color: 'var(--ink-58)', fontWeight: 500 }}>Savings rate</div>
            <div style={{ fontSize: 12, color: 'var(--ink-52)', marginTop: 4 }}>Share of income you kept this month.</div>
          </div>
        </div>
      </div>

      <div className="grid grid-3" style={{ marginBottom: 18 }}>
        <Card title="Spending pace" className="span-2">
          {s ? (
            <>
              <TrendChart
                labels={s.daily.spent.map((_, i) => `Day ${i + 1}`)}
                series={[
                  {
                    name: 'Spent so far',
                    // The current month stops at today rather than drawing a flat line into the future.
                    values: month === thisMonth() ? s.daily.spent.slice(0, new Date().getDate()) : s.daily.spent,
                    color: 'var(--series-1)',
                    area: true,
                  },
                  { name: 'Even pace', values: s.daily.pace, color: 'var(--ink-42)', dash: '5 5' },
                ]}
                format={(n) => money(n, true)}
                height={230}
              />
              <Legend series={[{ name: 'Spent so far', color: 'var(--series-1)' }, { name: 'Even pace for your budget', color: 'var(--ink-42)', dash: '5 5' }]} />
            </>
          ) : (
            <Skeleton height={230} />
          )}
        </Card>
        <QuickAdd categories={(s?.categories ?? []).map((c) => c.name)} onAdded={reload} />
      </div>

      <div className="grid grid-2" style={{ marginBottom: 18 }}>
        <Categories summary={s} onChange={reload} />
        <Goals goals={s?.goals ?? []} onChange={summary.reload} />
      </div>

      <Card title="Transactions" action={<span className="chip">{entries.data?.length ?? 0} this month</span>}>
        {entries.data && entries.data.length ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Category</th>
                  <th>Note</th>
                  <th className="r">Amount</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {entries.data.map((e) => (
                  <tr key={e.id}>
                    <td className="num">{e.occurred_on}</td>
                    <td>{e.category || <span className="text-muted">—</span>}</td>
                    <td style={{ color: 'var(--ink-62)' }}>{e.note}</td>
                    <td className={`r num ${e.direction === 'income' ? 'pos' : ''}`}>
                      {e.direction === 'income' ? '+' : '−'}
                      {money(e.amount)}
                    </td>
                    <td className="r">
                      <DeleteButton label={`Delete ${e.note || 'entry'}`} onDelete={() => api.deleteEntry(e.id)} onDone={reload} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-muted" style={{ margin: 0 }}>Nothing recorded for {monthLabel(month)} yet. Use Quick add above.</p>
        )}
      </Card>
    </div>
  );
}

function DeleteButton({ label, onDelete, onDone }: { label: string; onDelete: () => Promise<unknown>; onDone: () => void }) {
  const { busy, run } = useAction();
  return (
    <button
      type="button"
      className="icon-btn"
      aria-label={label}
      disabled={busy}
      onClick={async () => {
        await run(onDelete);
        onDone();
      }}
    >
      <Icon name="trash" size={16} />
    </button>
  );
}

function QuickAdd({ categories, onAdded }: { categories: string[]; onAdded: () => void }) {
  const [direction, setDirection] = useState<'expense' | 'income'>('expense');
  const [amount, setAmount] = useState('');
  const [category, setCategory] = useState('');
  const [note, setNote] = useState('');
  const [day, setDay] = useState(today());
  const { busy, run } = useAction();

  const submit = async () => {
    const value = Number(amount);
    if (!(value > 0)) return;
    const ok = await run(
      () => api.addEntry({ occurred_on: day, amount: value, direction, category: direction === 'income' ? 'Income' : category, note }),
      direction === 'income' ? `Income of ${money(value)} added` : `Spent ${money(value)}${category ? ` on ${category}` : ''}`,
    );
    if (ok) {
      setAmount('');
      setNote('');
      onAdded();
    }
  };

  return (
    <Card title="Quick add">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
        style={{ display: 'grid', gap: 12 }}
      >
        <div className="seg" role="radiogroup" aria-label="Type" style={{ width: '100%' }}>
          {(['expense', 'income'] as const).map((d) => (
            <label key={d} className="seg-opt" style={{ flex: 1, justifyContent: 'center' }}>
              <input type="radio" name="dir" checked={direction === d} onChange={() => setDirection(d)} />
              {d === 'expense' ? 'Expense' : 'Income'}
            </label>
          ))}
        </div>
        <div className="field">
          <label htmlFor="qa-amount">Amount</label>
          <input id="qa-amount" className="input num" inputMode="decimal" placeholder="0.00" value={amount} onChange={(e) => setAmount(e.target.value)} style={{ fontSize: 22, fontWeight: 600 }} />
        </div>
        {direction === 'expense' ? (
          <div className="row wrap-row" style={{ gap: 6 }}>
            {categories.length ? (
              categories.map((c) => (
                <button key={c} type="button" className={`chip ${category === c ? 'chip-accent' : ''}`} style={{ border: 0, cursor: 'pointer' }} onClick={() => setCategory(category === c ? '' : c)}>
                  {c}
                </button>
              ))
            ) : (
              <span className="text-muted" style={{ fontSize: 12.5 }}>Add budget categories to tag spending.</span>
            )}
          </div>
        ) : null}
        <div className="row" style={{ gap: 8 }}>
          <input className="input" placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
          <input className="input" type="date" value={day} onChange={(e) => setDay(e.target.value)} style={{ width: 150 }} aria-label="Date" />
        </div>
        <button className="btn btn-primary" type="submit" disabled={busy || !(Number(amount) > 0)}>
          <Icon name="plus" size={16} /> Add {direction}
        </button>
      </form>
    </Card>
  );
}

function Categories({ summary, onChange }: { summary: MonthSummary | null; onChange: () => void }) {
  const [name, setName] = useState('');
  const [limit, setLimit] = useState('');
  const [kind, setKind] = useState<CategoryKind>('need');
  const [editing, setEditing] = useState<string | null>(null);
  const [editValue, setEditValue] = useState('');
  const { busy, run } = useAction();
  const cats = summary?.categories ?? [];

  const save = async (n: string, l: number, k: CategoryKind, msg?: string) => {
    await run(() => api.saveCategory(n, l, k), msg);
    onChange();
  };

  return (
    <Card title="Budget categories" action={<span className="chip">{cats.length} categories</span>}>
      {!summary ? (
        <Skeleton height={160} />
      ) : cats.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '12px 0 18px' }}>
          <p className="text-muted">No categories yet. Start from a sensible template and adjust the limits.</p>
          <button
            className="btn btn-primary"
            type="button"
            disabled={busy}
            onClick={async () => {
              await run(async () => {
                for (const [n, l, k] of TEMPLATE) await api.saveCategory(n, l, k);
              }, 'Starter budget created');
              onChange();
            }}
          >
            <Icon name="sparkle" size={16} /> Use starter budget
          </button>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: 16 }}>
          {cats.map((c) => {
            const over = c.monthly_limit > 0 && c.spent > c.monthly_limit;
            return (
              <div key={c.name}>
                <ProgressBar
                  value={c.spent}
                  max={c.monthly_limit || Math.max(c.spent, 1)}
                  tone={usageTone(c.spent, c.monthly_limit)}
                  label={
                    <span className="row" style={{ gap: 8 }}>
                      <b style={{ fontWeight: 600 }}>{c.name}</b>
                      <span className={KIND_CHIP[c.kind] ?? 'chip'} style={{ fontSize: 10.5, padding: '1px 7px' }}>{c.kind}</span>
                      {over ? <span className="chip chip-neg" style={{ fontSize: 10.5, padding: '1px 7px' }}>over by {money(c.spent - c.monthly_limit, true)}</span> : null}
                    </span>
                  }
                  right={
                    editing === c.name ? (
                      <form
                        className="row"
                        style={{ gap: 4 }}
                        onSubmit={async (e) => {
                          e.preventDefault();
                          setEditing(null);
                          await save(c.name, Number(editValue) || 0, c.kind, `${c.name} limit updated`);
                        }}
                      >
                        <input className="input num" autoFocus value={editValue} onChange={(e) => setEditValue(e.target.value)} style={{ width: 90, minHeight: 28, padding: '2px 8px' }} aria-label={`${c.name} limit`} />
                        <button className="icon-btn" type="submit" aria-label="Save"><Icon name="check" size={15} /></button>
                      </form>
                    ) : (
                      <button
                        type="button"
                        className="btn-ghost btn btn-sm"
                        title="Click to edit the limit"
                        onClick={() => {
                          setEditing(c.name);
                          setEditValue(String(c.monthly_limit));
                        }}
                      >
                        {money(c.spent, true)} / {c.monthly_limit ? money(c.monthly_limit, true) : 'set limit'}
                      </button>
                    )
                  }
                />
              </div>
            );
          })}
        </div>
      )}

      <form
        className="row wrap-row"
        style={{ gap: 8, marginTop: 18, paddingTop: 14, borderTop: '1px solid var(--color-divider)' }}
        onSubmit={async (e) => {
          e.preventDefault();
          if (!name.trim()) return;
          await save(name.trim(), Number(limit) || 0, kind, `Category ${name.trim()} added`);
          setName('');
          setLimit('');
        }}
      >
        <input className="input" placeholder="New category" value={name} onChange={(e) => setName(e.target.value)} style={{ flex: '2 1 140px' }} />
        <input className="input num" placeholder="Limit" inputMode="decimal" value={limit} onChange={(e) => setLimit(e.target.value)} style={{ flex: '1 1 80px' }} />
        <select className="input" value={kind} onChange={(e) => setKind(e.target.value as CategoryKind)} style={{ flex: '1 1 100px' }} aria-label="Kind">
          <option value="need">Need</option>
          <option value="want">Want</option>
          <option value="savings">Savings</option>
        </select>
        <button className="btn" type="submit" disabled={busy || !name.trim()}>
          <Icon name="plus" size={15} /> Add
        </button>
      </form>
    </Card>
  );
}

function monthsUntil(deadline: string | null): number | null {
  if (!deadline) return null;
  const d = new Date(deadline);
  const now = new Date();
  return Math.max(0, (d.getFullYear() - now.getFullYear()) * 12 + d.getMonth() - now.getMonth());
}

function GoalCard({ goal, onChange }: { goal: SavingsGoal; onChange: () => void }) {
  const [custom, setCustom] = useState('');
  const { busy, run } = useAction();
  const frac = goal.target > 0 ? goal.saved / goal.target : 0;
  const months = monthsUntil(goal.deadline);
  const remaining = Math.max(0, goal.target - goal.saved);
  const add = async (amount: number) => {
    await run(() => api.contribute(goal.id, amount), amount > 0 ? `${money(amount)} added to ${goal.name}` : undefined);
    onChange();
  };
  return (
    <div className="card card-hover" style={{ padding: 16, background: 'var(--color-surface)' }}>
      <div className="row" style={{ gap: 14, alignItems: 'flex-start' }}>
        <Ring value={frac} size={72} stroke={8} tone={frac >= 1 ? 'var(--pos)' : 'var(--color-accent)'}>
          <b className="num" style={{ fontSize: 14 }}>{pct(Math.min(frac, 9.99), 0)}</b>
        </Ring>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="row between">
            <b>{goal.name}</b>
            <DeleteButton label={`Delete ${goal.name}`} onDelete={() => api.deleteGoal(goal.id)} onDone={onChange} />
          </div>
          <div className="num" style={{ fontSize: 13, color: 'var(--ink-62)' }}>
            {money(goal.saved, true)} of {money(goal.target, true)}
          </div>
          <div style={{ fontSize: 12, color: 'var(--ink-52)', marginTop: 2 }}>
            {frac >= 1
              ? '🎉 Goal reached'
              : months !== null
                ? months > 0
                  ? `${money(remaining / months, true)}/month to hit ${goal.deadline}`
                  : `Deadline ${goal.deadline} — ${money(remaining, true)} to go`
                : `${money(remaining, true)} to go`}
          </div>
        </div>
      </div>
      <div className="row wrap-row" style={{ gap: 6, marginTop: 12 }}>
        {[25, 100, 250].map((a) => (
          <button key={a} type="button" className="btn btn-sm" disabled={busy} onClick={() => add(a)}>
            +{a}
          </button>
        ))}
        <form
          className="row"
          style={{ gap: 4, flex: 1, minWidth: 120 }}
          onSubmit={async (e) => {
            e.preventDefault();
            const v = Number(custom);
            if (v) {
              await add(v);
              setCustom('');
            }
          }}
        >
          <input className="input num" placeholder="± amount" value={custom} onChange={(e) => setCustom(e.target.value)} style={{ minHeight: 30, padding: '3px 8px' }} aria-label={`Custom amount for ${goal.name}`} />
        </form>
      </div>
    </div>
  );
}

function Goals({ goals, onChange }: { goals: SavingsGoal[]; onChange: () => void }) {
  const [name, setName] = useState('');
  const [target, setTarget] = useState('');
  const [deadline, setDeadline] = useState('');
  const { busy, run } = useAction();
  return (
    <Card title="Savings goals" action={<span className="chip chip-pos">{goals.filter((g) => g.saved >= g.target).length} reached</span>}>
      <div style={{ display: 'grid', gap: 12 }}>
        {goals.map((g) => (
          <GoalCard key={g.id} goal={g} onChange={onChange} />
        ))}
        {goals.length === 0 ? <p className="text-muted" style={{ margin: 0 }}>Set a goal — an emergency fund is a good first one.</p> : null}
      </div>
      <form
        className="row wrap-row"
        style={{ gap: 8, marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--color-divider)' }}
        onSubmit={async (e) => {
          e.preventDefault();
          const t = Number(target);
          if (!name.trim() || !(t > 0)) return;
          await run(() => api.addGoal(name.trim(), t, 0, deadline || null), `Goal ${name.trim()} created`);
          setName('');
          setTarget('');
          setDeadline('');
          onChange();
        }}
      >
        <input className="input" placeholder="Goal name" value={name} onChange={(e) => setName(e.target.value)} style={{ flex: '2 1 140px' }} />
        <input className="input num" placeholder="Target" inputMode="decimal" value={target} onChange={(e) => setTarget(e.target.value)} style={{ flex: '1 1 90px' }} />
        <input className="input" type="date" value={deadline} onChange={(e) => setDeadline(e.target.value)} style={{ flex: '1 1 140px' }} aria-label="Deadline" />
        <button className="btn" type="submit" disabled={busy || !name.trim()}>
          <Icon name="target" size={15} /> Add goal
        </button>
      </form>
    </Card>
  );
}
