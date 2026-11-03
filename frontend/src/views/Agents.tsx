// ═══════════════════════════════════════════════════════
// FinSight web — Agents
// ═══════════════════════════════════════════════════════
//
// Four reinforcement-learning agents, one lab each. The agent cards are the
// tabs. The selected tab is remembered per browser so a reload lands back on
// the lab you were in.
//
//   portfolio  policy gradient over target weights → paper-order proposals
//   budget     contextual bandit over savings rates → category limits
//   timing     policy gradient over long / flat → signals only
//   alerts     Thompson-sampling bandit over alert features → a ranking
// ═══════════════════════════════════════════════════════

import { useState } from 'react';
import * as api from '../api/client';
import { Disclaimer, Icon, PageTitle } from '../components/ui';
import type { View } from '../hooks/useRoute';
import { useResource } from '../hooks/useResource';
import { AlertLab } from './agents/AlertLab';
import { BudgetLab } from './agents/BudgetLab';
import { PortfolioLab } from './agents/PortfolioLab';
import { TimingLab } from './agents/TimingLab';

export { SELECTED_RUN_KEY } from './agents/PortfolioLab';
export { StatusChip } from './agents/shared';

type Tab = 'portfolio' | 'budget' | 'timing' | 'alerts';
const TAB_KEY = 'finsight.agentTab';
const ICONS: Record<Tab, string> = { portfolio: 'chart', budget: 'wallet', timing: 'bolt', alerts: 'inbox' };

function storedTab(): Tab {
  try {
    const v = localStorage.getItem(TAB_KEY);
    if (v === 'portfolio' || v === 'budget' || v === 'timing' || v === 'alerts') return v;
  } catch {
    /* storage blocked */
  }
  return 'portfolio';
}

export function Agents({ navigate }: { navigate: (v: View) => void }) {
  const agents = useResource(() => api.listAgents(), []);
  const [tab, setTabState] = useState<Tab>(storedTab);
  const setTab = (t: Tab) => {
    setTabState(t);
    try {
      localStorage.setItem(TAB_KEY, t);
    } catch {
      /* storage blocked: the choice just is not remembered */
    }
  };

  return (
    <div className="page">
      <PageTitle
        title="Agents"
        subtitle="Four reinforcement-learning agents. Train one, judge it on data it never saw, then decide whether to act on it."
      />

      <div className="grid grid-4" role="tablist" aria-label="Agents" style={{ marginBottom: 22 }}>
        {(agents.data ?? []).map((a) => {
          const id = a.id as Tab;
          const selected = tab === id;
          return (
            <button
              key={a.id}
              type="button"
              role="tab"
              aria-selected={selected}
              onClick={() => setTab(id)}
              className="card card-hover rise"
              style={{
                padding: 16,
                textAlign: 'left',
                font: 'inherit',
                color: 'inherit',
                cursor: 'pointer',
                borderColor: selected ? 'var(--color-accent)' : undefined,
                boxShadow: selected ? '0 0 0 3px color-mix(in srgb, var(--color-accent) 20%, transparent), var(--shadow-md)' : undefined,
                background: selected ? 'color-mix(in srgb, var(--color-accent) 5%, var(--color-surface))' : undefined,
              }}
            >
              <div className="row between" style={{ marginBottom: 8 }}>
                <span
                  style={{
                    display: 'grid',
                    placeItems: 'center',
                    width: 34,
                    height: 34,
                    borderRadius: 10,
                    background: selected ? 'var(--color-accent)' : 'var(--color-accent-100)',
                    color: selected ? '#fff' : 'var(--color-accent)',
                    transition: 'background 0.2s ease',
                  }}
                >
                  <Icon name={ICONS[id] ?? 'sparkle'} size={18} />
                </span>
                <span className="chip chip-pos">Live</span>
              </div>
              <b>{a.name}</b>
              <p style={{ fontSize: 12.5, color: 'var(--ink-58)', margin: '4px 0 6px' }}>{a.summary}</p>
              <span style={{ fontSize: 11.5, color: 'var(--ink-45)' }}>{a.method}</span>
            </button>
          );
        })}
      </div>

      <div role="tabpanel" key={tab} className="rise">
        {tab === 'portfolio' ? <PortfolioLab navigate={navigate} /> : null}
        {tab === 'budget' ? <BudgetLab /> : null}
        {tab === 'timing' ? <TimingLab /> : null}
        {tab === 'alerts' ? <AlertLab navigate={navigate} /> : null}
      </div>

      <Disclaimer />
    </div>
  );
}
