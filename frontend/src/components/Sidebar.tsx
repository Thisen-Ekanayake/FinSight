// ═══════════════════════════════════════════════════════
// FinSight web — sidebar navigation
// ═══════════════════════════════════════════════════════
// Two groups — Money (the finance platform) and Research (the original
// research + monitoring desk) — plus theme and sign-out at the foot. Collapses
// into a horizontal scroller on narrow screens (tokens.css).
// ═══════════════════════════════════════════════════════

import { Icon } from './ui';
import type { ThemeSetting } from '../hooks/useTheme';
import type { View } from '../hooks/useRoute';

const GROUPS: { label: string; items: { view: View; label: string; icon: string }[] }[] = [
  {
    label: 'Money',
    items: [
      { view: 'overview', label: 'Overview', icon: 'home' },
      { view: 'budget', label: 'Budget', icon: 'wallet' },
      { view: 'portfolio', label: 'Portfolio', icon: 'chart' },
      { view: 'agents', label: 'Agents', icon: 'brain' },
    ],
  },
  {
    label: 'Research',
    items: [
      { view: 'desk', label: 'Desk', icon: 'inbox' },
      { view: 'ask', label: 'Ask', icon: 'search' },
      { view: 'findings', label: 'Findings', icon: 'flag' },
      { view: 'watchlist', label: 'Watchlist', icon: 'eye' },
      { view: 'system', label: 'System', icon: 'gear' },
    ],
  },
];

export function Sidebar({
  view,
  navigate,
  pendingCount,
  proposalCount,
  theme,
  setTheme,
  email,
  onSignOut,
}: {
  view: View;
  navigate: (view: View) => void;
  pendingCount: number;
  proposalCount: number;
  theme: ThemeSetting;
  setTheme: (theme: ThemeSetting) => void;
  email?: string;
  onSignOut?: () => void;
}) {
  const badge = (v: View) => (v === 'desk' ? pendingCount : v === 'portfolio' ? proposalCount : 0);

  return (
    <aside className="sidebar" aria-label="Navigation">
      <div className="brand">
        <span className="brand-mark">F</span>
        <span className="brand-name" style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: 18 }}>
          FinSight
        </span>
      </div>

      {GROUPS.map((group) => (
        <nav key={group.label} aria-label={group.label} style={{ display: 'contents' }}>
          <div className="nav-group">{group.label}</div>
          {group.items.map((item) => {
            const count = badge(item.view);
            return (
              <button
                key={item.view}
                type="button"
                className="nav-item"
                aria-current={view === item.view ? 'page' : undefined}
                onClick={() => navigate(item.view)}
              >
                <Icon name={item.icon} size={18} />
                <span>{item.label}</span>
                {count > 0 ? (
                  <span className="badge" aria-label={`${count} waiting`}>
                    {count}
                  </span>
                ) : null}
              </button>
            );
          })}
        </nav>
      ))}

      <div className="sidebar-foot" style={{ marginTop: 'auto', display: 'grid', gap: 10, paddingTop: 18 }}>
        <div className="seg" role="radiogroup" aria-label="Theme" style={{ width: '100%' }}>
          {(['light', 'auto', 'dark'] as ThemeSetting[]).map((option) => (
            <label key={option} className="seg-opt" style={{ flex: 1, justifyContent: 'center', padding: '5px 6px', fontSize: 12 }}>
              <input type="radio" name="fs-theme" checked={theme === option} onChange={() => setTheme(option)} />
              {option === 'light' ? <Icon name="sun" size={14} /> : option === 'dark' ? <Icon name="moon" size={14} /> : 'Auto'}
            </label>
          ))}
        </div>
        {email ? (
          <div className="row between" style={{ gap: 8, padding: '8px 6px 0', borderTop: '1px solid var(--color-divider)' }}>
            <span
              title={email}
              style={{ fontSize: 12.5, color: 'var(--ink-58)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            >
              {email}
            </span>
            <button type="button" className="icon-btn" onClick={onSignOut} aria-label="Sign out" title="Sign out">
              <Icon name="logout" size={16} />
            </button>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
