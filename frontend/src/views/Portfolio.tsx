// ═══════════════════════════════════════════════════════
// FinSight web — Portfolio (Alpaca PAPER)
// ═══════════════════════════════════════════════════════
//
// Link a paper account, see what it holds, and decide the agent's proposed
// orders one by one. Every row defaults to "skip": an untouched form submits
// nothing, the same principle the Desk applies to HIGH alerts. Submitting
// takes two clicks — the second confirms the count.
// ═══════════════════════════════════════════════════════

import { useEffect, useMemo, useState } from 'react';
import * as api from '../api/client';
import type { BrokerStatus, OrderProposal } from '../api/types';
import { Card, Delta, Disclaimer, Icon, PageTitle, ProgressBar, Skeleton, Stat, Toggle, useAction } from '../components/ui';
import type { View } from '../hooks/useRoute';
import { useResource } from '../hooks/useResource';
import { money, pct, signedMoney, signedPct } from '../lib/money';
import { Donut, SERIES } from '../viz/finance';
import { SELECTED_RUN_KEY, StatusChip } from './Agents';

type Choice = 'approve' | 'reject' | 'skip';

export function Portfolio({ navigate }: { navigate: (v: View) => void }) {
  const status = useResource(() => api.brokerStatus(), [], { pollMs: 30_000 });
  const proposals = useResource(() => api.listProposals(), []);
  const runs = useResource(() => api.listPolicyRuns(20, 'portfolio'), []);
  const s = status.data;

  return (
    <div className="page">
      <PageTitle
        title="Portfolio"
        subtitle="Your Alpaca paper account: real market prices, simulated money. The agent proposes; you decide."
        action={s?.linked ? <span className="chip chip-pos"><Icon name="link" size={13} /> Paper ···{s.key_hint}</span> : null}
      />

      {!s ? (
        <Skeleton height={220} />
      ) : !s.linked ? (
        <LinkCard status={s} onLinked={status.reload} />
      ) : (
        <>
          <AccountStats status={s} />
          <div className="grid grid-3" style={{ marginBottom: 18 }}>
            <Card className="span-2" title="Positions">
              <Positions status={s} />
            </Card>
            <Safety status={s} onChange={status.reload} />
          </div>
          <Proposals
            proposals={proposals.data ?? []}
            runs={(runs.data ?? []).filter((r) => r.status === 'DONE')}
            limits={s.limits}
            paused={s.paused}
            onChange={() => {
              proposals.reload();
              status.reload();
            }}
            navigate={navigate}
          />
        </>
      )}

      <div style={{ marginTop: 18 }}>
        <Disclaimer />
      </div>
    </div>
  );
}

function LinkCard({ status, onLinked }: { status: BrokerStatus; onLinked: () => void }) {
  const [keyId, setKeyId] = useState('');
  const [secret, setSecret] = useState('');
  const { busy, run } = useAction();
  const live = keyId.trim().toUpperCase().startsWith('AK');

  return (
    <div className="grid grid-2">
      <div className="hero rise">
        <span className="chip" style={{ background: 'rgba(255,255,255,0.18)', color: '#fff' }}>
          <Icon name="shield" size={13} /> Paper trading only
        </span>
        <h2 style={{ margin: '14px 0 8px', color: '#fff' }}>Practise with real prices, not real money</h2>
        <p style={{ opacity: 0.9, maxWidth: '48ch' }}>
          Alpaca is a US broker that offers free <b>paper</b> accounts: a $100k virtual balance that fills orders at live market
          prices. FinSight only ever talks to the paper API, and it refuses live keys outright.
        </p>
        <ol style={{ margin: '12px 0 0', paddingLeft: 18, lineHeight: 1.9, opacity: 0.95 }}>
          <li>Create a free account at alpaca.markets</li>
          <li>Switch to your <b>Paper</b> account in the dashboard</li>
          <li>Generate API keys — the key id starts with <span className="mono">PK</span></li>
          <li>Paste them here. They're verified, then stored encrypted.</li>
        </ol>
      </div>
      <Card title="Link Alpaca paper account">
        {!status.encryption_available ? (
          <p className="chip chip-warn" style={{ whiteSpace: 'normal', padding: '8px 12px' }}>
            Linking is disabled on this server until the operator sets BROKER_ENCRYPTION_KEY — keys are never stored unencrypted.
          </p>
        ) : null}
        <form
          style={{ display: 'grid', gap: 14 }}
          onSubmit={async (e) => {
            e.preventDefault();
            const ok = await run(() => api.linkBroker(keyId, secret), 'Paper account linked');
            if (ok) {
              setSecret('');
              onLinked();
            }
          }}
        >
          <div className="field">
            <label htmlFor="key-id">API key id</label>
            <input id="key-id" className="input mono" autoComplete="off" placeholder="PK…" value={keyId} onChange={(e) => setKeyId(e.target.value)} />
            {live ? <div className="chip chip-neg" style={{ marginTop: 6 }}>That's a LIVE key (AK…). Use a paper key (PK…).</div> : null}
          </div>
          <div className="field">
            <label htmlFor="secret">Secret key</label>
            <input id="secret" className="input mono" type="password" autoComplete="off" value={secret} onChange={(e) => setSecret(e.target.value)} />
          </div>
          <button className="btn btn-primary" type="submit" disabled={busy || live || !keyId || !secret || !status.encryption_available}>
            <Icon name="link" size={16} /> {busy ? 'Verifying…' : 'Verify & link'}
          </button>
        </form>
      </Card>
    </div>
  );
}

function AccountStats({ status }: { status: BrokerStatus }) {
  const a = status.account;
  const dayChange = a && a.last_equity ? a.equity - a.last_equity : 0;
  const invested = status.positions.reduce((t, p) => t + p.market_value, 0);
  if (status.error) return <p className="chip chip-neg" style={{ whiteSpace: 'normal', marginBottom: 18 }}>{status.error}</p>;
  return (
    <div className="grid grid-4" style={{ marginBottom: 18 }}>
      <Stat label="Equity" value={a?.equity ?? 0} format={(n) => money(n, true)} icon="wallet"
        footer={a?.last_equity ? <Delta value={dayChange} format={(n) => `${signedMoney(n, true)} today`} /> : null} />
      <Stat label="Cash" value={a?.cash ?? 0} format={(n) => money(n, true)} icon="inbox" tone="var(--series-2)" />
      <Stat label="Invested" value={invested} format={(n) => money(n, true)} icon="chart" tone="var(--pos)"
        footer={a?.equity ? `${pct(invested / a.equity, 0)} of equity` : null} />
      <Stat label="Positions" value={status.positions.length} format={(n) => String(Math.round(n))} icon="target" tone="var(--series-3)" />
    </div>
  );
}

function Positions({ status }: { status: BrokerStatus }) {
  const cash = Math.max(0, status.account?.cash ?? 0);
  const slices = [
    ...status.positions.map((p, i) => ({ label: p.symbol, value: p.market_value, color: SERIES[i % SERIES.length] })),
    ...(cash > 0 ? [{ label: 'Cash', value: cash, color: 'var(--ink-42)' }] : []),
  ];
  if (!status.positions.length) {
    return <p className="text-muted" style={{ margin: 0 }}>No positions yet — the account is all cash. Approve some agent proposals below to invest.</p>;
  }
  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr)' }}>
      <Donut data={slices} size={160} center={<span style={{ fontSize: 12, color: 'var(--ink-52)' }}>Hover a slice</span>} />
      <div style={{ overflowX: 'auto' }}>
        <table className="table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th className="r">Qty</th>
              <th className="r">Price</th>
              <th className="r">Value</th>
              <th className="r">P/L</th>
            </tr>
          </thead>
          <tbody>
            {status.positions.map((p) => (
              <tr key={p.symbol}>
                <td><b>{p.symbol}</b></td>
                <td className="r num">{p.qty.toFixed(4)}</td>
                <td className="r num">{money(p.current_price)}</td>
                <td className="r num">{money(p.market_value)}</td>
                <td className={`r num ${p.unrealized_pl >= 0 ? 'pos' : 'neg'}`}>
                  {signedMoney(p.unrealized_pl)} ({signedPct(p.unrealized_plpc)})
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Safety({ status, onChange }: { status: BrokerStatus; onChange: () => void }) {
  const { busy, run } = useAction();
  const l = status.limits;
  return (
    <Card title="Safety controls">
      <div style={{ display: 'grid', gap: 14 }}>
        {l.kill_switch ? (
          <span className="chip chip-neg" style={{ whiteSpace: 'normal', padding: '8px 12px' }}>
            Server kill switch is ON — no orders can be submitted by anyone.
          </span>
        ) : null}
        <Toggle
          checked={status.paused}
          disabled={busy}
          onChange={async (v) => {
            await run(() => api.pauseTrading(v), v ? 'Trading paused for your account' : 'Trading resumed');
            onChange();
          }}
          label={<span><b>Pause my trading</b><br /><span style={{ fontSize: 12, color: 'var(--ink-52)' }}>Your personal kill switch.</span></span>}
        />
        <div style={{ display: 'grid', gap: 8, fontSize: 13 }}>
          <div className="row between"><span className="text-muted">Max per order</span><b className="num">{money(l.max_order_notional, true)}</b></div>
          <div className="row between"><span className="text-muted">Max per position</span><b className="num">{pct(l.max_position_pct, 0)}</b></div>
          <div className="row between"><span className="text-muted">Orders per day</span><b className="num">{l.max_orders_per_day}</b></div>
        </div>
        <button
          className="btn btn-danger btn-sm"
          type="button"
          disabled={busy}
          onClick={async () => {
            await run(() => api.unlinkBroker(), 'Paper account unlinked and keys deleted');
            onChange();
          }}
        >
          Unlink & delete keys
        </button>
      </div>
    </Card>
  );
}

function Proposals({
  proposals,
  runs,
  limits,
  paused,
  onChange,
  navigate,
}: {
  proposals: OrderProposal[];
  runs: { run_id: string; tickers: string[]; created_at: string | null }[];
  limits: BrokerStatus['limits'];
  paused: boolean;
  onChange: () => void;
  navigate: (v: View) => void;
}) {
  const [runId, setRunId] = useState<string>('');
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const [confirming, setConfirming] = useState(false);
  const { busy, run } = useAction();

  useEffect(() => {
    if (runId || !runs.length) return;
    let preferred: string | null = null;
    try {
      preferred = sessionStorage.getItem(SELECTED_RUN_KEY);
    } catch {
      preferred = null;
    }
    setRunId(runs.find((r) => r.run_id === preferred)?.run_id ?? runs[0].run_id);
  }, [runs, runId]);

  const pending = proposals.filter((p) => p.status === 'PENDING');
  const history = proposals.filter((p) => p.status !== 'PENDING').slice(0, 15);
  const approve = pending.filter((p) => choices[p.proposal_id] === 'approve');
  const reject = pending.filter((p) => choices[p.proposal_id] === 'reject');
  const buyTotal = approve.filter((p) => p.side === 'buy').reduce((t, p) => t + p.notional, 0);
  const sellTotal = approve.filter((p) => p.side === 'sell').reduce((t, p) => t + p.notional, 0);
  const blocked = limits.kill_switch || paused;

  useEffect(() => {
    if (!confirming) return;
    const t = window.setTimeout(() => setConfirming(false), 5000);
    return () => window.clearTimeout(t);
  }, [confirming]);

  const setAll = (c: Choice) => setChoices(Object.fromEntries(pending.map((p) => [p.proposal_id, c])));

  const runLabel = useMemo(
    () => (r: { tickers: string[]; created_at: string | null }) =>
      `${r.tickers.join(', ')} · ${r.created_at ? new Date(r.created_at).toLocaleDateString() : ''}`,
    [],
  );

  return (
    <Card
      title={
        <div>
          <h3 style={{ margin: 0 }}>Agent proposals</h3>
          <div style={{ fontSize: 12.5, color: 'var(--ink-52)' }}>Nothing is sent until you approve it. Rows default to skip.</div>
        </div>
      }
      action={pending.length ? <span className="chip chip-accent">{pending.length} waiting</span> : null}
    >
      <div className="row wrap-row" style={{ gap: 8, marginBottom: 16 }}>
        {runs.length ? (
          <>
            <select className="input" value={runId} onChange={(e) => setRunId(e.target.value)} style={{ flex: '1 1 260px', maxWidth: 420 }} aria-label="Training run">
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>{runLabel(r)}</option>
              ))}
            </select>
            <button
              className="btn btn-primary"
              type="button"
              disabled={busy || !runId}
              onClick={async () => {
                const out = await run(() => api.proposeOrders(runId));
                if (out) {
                  setChoices({});
                  onChange();
                }
              }}
            >
              <Icon name="sparkle" size={16} /> Generate proposals
            </button>
          </>
        ) : (
          <button className="btn" type="button" onClick={() => navigate('agents')}>
            <Icon name="brain" size={16} /> Train an agent first
          </button>
        )}
      </div>

      {pending.length ? (
        <>
          <div className="row between wrap-row" style={{ marginBottom: 8, gap: 8 }}>
            <span className="text-muted" style={{ fontSize: 12.5 }}>Sells are listed first so their proceeds fund the buys.</span>
            <div className="row" style={{ gap: 6 }}>
              <button type="button" className="btn btn-sm" onClick={() => setAll('approve')}>Approve all</button>
              <button type="button" className="btn btn-sm" onClick={() => setAll('reject')}>Reject all</button>
              <button type="button" className="btn btn-sm btn-ghost" onClick={() => setChoices({})}>Reset</button>
            </div>
          </div>
          <div style={{ display: 'grid', gap: 10 }}>
            {pending.map((p) => {
              const choice = choices[p.proposal_id] ?? 'skip';
              return (
                <div
                  key={p.proposal_id}
                  className="card"
                  style={{
                    padding: 14,
                    borderColor: choice === 'approve' ? 'var(--pos)' : choice === 'reject' ? 'var(--neg)' : undefined,
                    transition: 'border-color 0.15s ease',
                  }}
                >
                  <div className="row between wrap-row" style={{ gap: 12 }}>
                    <div className="row" style={{ gap: 10, minWidth: 180 }}>
                      <span className={`chip ${p.side === 'buy' ? 'chip-pos' : 'chip-neg'}`}>
                        <Icon name={p.side === 'buy' ? 'up' : 'down'} size={12} /> {p.side.toUpperCase()}
                      </span>
                      <b style={{ fontSize: 16 }}>{p.symbol}</b>
                      <b className="num">{money(p.notional)}</b>
                    </div>
                    <div style={{ flex: '1 1 220px', minWidth: 180 }}>
                      <ProgressBar
                        value={p.target_weight}
                        max={Math.max(p.target_weight, p.current_weight, 0.01)}
                        tone={p.side === 'buy' ? 'var(--pos)' : 'var(--neg)'}
                        height={6}
                        label={<span style={{ fontSize: 12 }}>Weight {pct(p.current_weight)} → {pct(p.target_weight)}</span>}
                      />
                    </div>
                    <div className="seg" role="radiogroup" aria-label={`Decision for ${p.symbol}`}>
                      {(['approve', 'skip', 'reject'] as Choice[]).map((c) => (
                        <label key={c} className="seg-opt" style={{ padding: '5px 10px', fontSize: 12.5 }}>
                          <input type="radio" name={`d-${p.proposal_id}`} checked={choice === c} onChange={() => setChoices({ ...choices, [p.proposal_id]: c })} />
                          {c === 'approve' ? 'Approve' : c === 'reject' ? 'Reject' : 'Skip'}
                        </label>
                      ))}
                    </div>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--ink-52)', marginTop: 8 }}>{p.reason}</div>
                </div>
              );
            })}
          </div>
          <div className="row between wrap-row" style={{ marginTop: 16, gap: 12 }}>
            <span style={{ fontSize: 13, color: 'var(--ink-62)' }}>
              {approve.length} to submit ({money(buyTotal, true)} buys, {money(sellTotal, true)} sells) · {reject.length} to reject
            </span>
            <button
              className={`btn ${confirming ? 'btn-pos' : 'btn-primary'}`}
              type="button"
              disabled={busy || (approve.length === 0 && reject.length === 0) || (approve.length > 0 && blocked)}
              onClick={async () => {
                if (approve.length && !confirming) {
                  setConfirming(true);
                  return;
                }
                setConfirming(false);
                const out = await run(
                  () => api.decideProposals(approve.map((p) => p.proposal_id), reject.map((p) => p.proposal_id)),
                  approve.length ? `${approve.length} paper order(s) sent` : 'Decisions recorded',
                );
                if (out) {
                  setChoices({});
                  onChange();
                }
              }}
            >
              <Icon name="check" size={16} />
              {blocked && approve.length
                ? 'Trading is paused'
                : confirming
                  ? `Confirm ${approve.length} paper order(s)`
                  : approve.length
                    ? `Submit ${approve.length} to paper`
                    : 'Record rejections'}
            </button>
          </div>
        </>
      ) : (
        <p className="text-muted" style={{ margin: 0 }}>No proposals waiting. Pick a finished training run and generate a fresh set.</p>
      )}

      {history.length ? (
        <div style={{ marginTop: 22 }}>
          <div className="kicker" style={{ marginBottom: 8 }}>Recent decisions</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Order</th>
                  <th className="r">Amount</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {history.map((p) => (
                  <tr key={p.proposal_id}>
                    <td className="num">{p.decided_at ? new Date(p.decided_at).toLocaleString() : '—'}</td>
                    <td>{p.side.toUpperCase()} {p.symbol}</td>
                    <td className="r num">{money(p.notional)}</td>
                    <td title={p.error || undefined}><StatusChip status={p.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
