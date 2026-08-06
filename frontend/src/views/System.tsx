// ═══════════════════════════════════════════════════════
// FinSight web — System
// ═══════════════════════════════════════════════════════
//
// Purpose : The page you open when an answer looks thin.
//
// ══ A SPENT ALLOWANCE IS NOT A FAULT ══
//   Every data source is a free tier with a daily cap. When one runs dry the
//   features that depend on it quietly degrade — an answer comes back with no
//   press coverage folded in. That reads as the system being broken unless
//   the interface says otherwise, so an exhausted provider gets a sentence
//   about what stops working, not a red badge.
//
//   The old design had this twice: a Home page and a Health page both showing
//   a row of status metrics, neither actionable on a normal day. There is one
//   now, and the Desk carries the only status worth interrupting anyone for.
// ═══════════════════════════════════════════════════════

import * as api from '../api/client';
import type { Budget } from '../api/types';
import { useResource } from '../hooks/useResource';
import { ErrorNote, Kicker, Loading, Meter, PageHead } from '../components/primitives';
import { clock, percent } from '../lib/format';

export function System() {
  const health = useResource(() => api.health(), [], { pollMs: 30_000 });
  const budgets = useResource(() => api.budgets(), [], { pollMs: 60_000 });
  const config = useResource(() => api.config(), []);

  const services = health.data
    ? [
        { name: 'API', value: health.data.status === 'ok' ? 'up' : health.data.status, ok: health.data.status === 'ok' },
        { name: 'Database', value: health.data.database ? 'up' : 'down', ok: health.data.database },
        { name: 'Checkpointer', value: health.data.checkpointer ? 'up' : 'down', ok: health.data.checkpointer },
        { name: 'Vector store', value: health.data.qdrant_detail, ok: health.data.qdrant },
        {
          name: 'Scheduler',
          value: health.data.scheduler_enabled
            ? `on · next ${clock(health.data.scheduler_next_run_at)}`
            : 'off — cycles run on request',
          ok: true,
        },
      ]
    : [];

  // `exhausted` is true for a provider with limit 0 — which means "no cap
  // configured", not "spent". Saying an allowance ran out when none exists
  // would send someone looking for a problem that is not there, so the same
  // uncapped test BudgetRow applies is applied here.
  const anyExhausted = (budgets.data ?? []).some((b) => b.exhausted && b.limit > 0);
  const allUp = health.data ? health.data.database && health.data.checkpointer && health.data.qdrant : false;

  return (
    <main style={{ maxWidth: 760, margin: '0 auto', padding: '72px 40px 120px' }}>
      <PageHead title="Under the hood">
        {health.error
          ? 'The backend is not answering. Nothing below can be trusted until it does.'
          : allUp
            ? anyExhausted
              ? "Everything is up. One allowance is spent, so some answers will arrive incomplete — that is a cap, not a fault."
              : 'Everything is up. You only need this page when an answer looks thin — usually a spent allowance rather than a fault.'
            : 'Something is degraded. A vector store that is down costs you the filings specialist; the numeric answers still work.'}
      </PageHead>

      <ErrorNote error={health.error} />

      {health.loading ? <Loading /> : null}

      {services.length > 0 ? (
        <>
          <Kicker>Services</Kicker>
          {services.map((service) => (
            <div
              key={service.name}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 16,
                padding: '11px 2px',
                borderTop: '1px solid var(--color-divider)',
              }}
            >
              <span style={{ fontSize: 14, color: 'var(--ink-58)' }}>{service.name}</span>
              <span className="mono" style={{ fontSize: 12.5, textAlign: 'right', color: service.ok ? 'var(--color-text)' : 'var(--alert-ink)' }}>
                {service.ok ? '' : '[!!] '}
                {service.value}
              </span>
            </div>
          ))}
        </>
      ) : null}

      <Kicker style={{ margin: '52px 0 6px' }}>Today's allowances</Kicker>
      <ErrorNote error={budgets.error} />
      {(budgets.data ?? []).length === 0 && !budgets.loading ? (
        <p style={{ fontSize: 14, color: 'var(--ink-58)' }}>
          Nothing has been spent today. Allowances appear here once a provider has been called.
        </p>
      ) : null}
      {(budgets.data ?? []).map((budget) => (
        <BudgetRow key={budget.provider} budget={budget} />
      ))}

      <Kicker style={{ margin: '52px 0 6px' }}>How it is set up</Kicker>
      <ErrorNote error={config.error} />
      {config.data ? (
        <>
          {[
            { k: 'Environment', v: config.data.environment },
            { k: 'Model', v: `${config.data.model_pro} · ${config.data.model_flash} (${config.data.llm_backend})` },
            { k: 'Embeddings', v: `${config.data.embedding_model} · ${config.data.embedding_dim}d` },
            { k: 'Vector store', v: config.data.qdrant_url },
            { k: 'Run interval', v: `every ${config.data.monitor_cadence_hours} h` },
            { k: 'Fold threshold', v: String(config.data.dedup_tau_high) },
            { k: 'Second-look threshold', v: String(config.data.dedup_tau_low) },
            { k: 'HIGH needs approval', v: 'yes — always' },
            { k: 'Dispatch to', v: config.data.notification_sinks.join(', ') || 'nothing configured' },
            { k: 'Numeric tolerance', v: percent(config.data.numeric_tolerance, 2) },
            { k: 'Repair attempts', v: String(config.data.max_repair_attempts) },
            { k: 'Qualitative claims verified', v: config.data.verify_qualitative_claims ? 'yes' : 'no' },
            { k: 'Tracing', v: config.data.tracing_enabled ? `on · ${config.data.langsmith_project}` : 'off' },
          ].map((row) => (
            <div
              key={row.k}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: 16,
                padding: '11px 2px',
                borderTop: '1px solid var(--color-divider)',
              }}
            >
              <span style={{ fontSize: 14, color: 'var(--ink-58)' }}>{row.k}</span>
              <span className="mono" style={{ fontSize: 12.5, textAlign: 'right' }}>
                {row.v}
              </span>
            </div>
          ))}
        </>
      ) : null}
    </main>
  );
}

function BudgetRow({ budget }: { budget: Budget }) {
  // A provider with limit 0 is not budgeted at all — reporting it as "0 of 0,
  // exhausted" would be a lie about a source that has no cap to hit.
  const uncapped = budget.limit <= 0;
  const used = uncapped ? 0 : budget.used / budget.limit;
  const tone = budget.exhausted && !uncapped ? 'var(--alert)' : 'var(--color-accent)';

  const note = uncapped
    ? `${budget.used} calls · no cap configured`
    : budget.exhausted
      ? `${budget.used} of ${budget.limit} — spent until midnight`
      : `${budget.used} of ${budget.limit} · ${budget.remaining} left${budget.soft_limit_reached ? ' · slowing down' : ''}`;

  return (
    <div style={{ padding: '15px 0', borderTop: '1px solid var(--color-divider)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 16, marginBottom: 8 }}>
        <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 15, letterSpacing: '0.05em', textTransform: 'uppercase' }}>
          {budget.provider}
        </span>
        <span className="mono" style={{ fontSize: 12, color: budget.exhausted && !uncapped ? 'var(--alert-ink)' : 'var(--ink-52)' }}>
          {note}
        </span>
      </div>
      <Meter pct={used} tone={tone} />
      {budget.exhausted && !uncapped ? (
        <p style={{ fontSize: 12.5, margin: '8px 0 0', color: 'var(--ink-62)', maxWidth: '60ch', textWrap: 'pretty' }}>
          Answers and cycles that would have used {budget.provider} will come back without that component until the
          allowance resets. They are incomplete, not wrong.
        </p>
      ) : null}
    </div>
  );
}
