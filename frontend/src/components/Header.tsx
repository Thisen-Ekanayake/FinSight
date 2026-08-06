// ═══════════════════════════════════════════════════════
// FinSight web — persistent header
// ═══════════════════════════════════════════════════════
//
// Five destinations and the theme control. The Desk badge carries the count
// of decisions waiting — the one number worth knowing from any screen, and
// the reason the old design's "go to page four" banner is gone.
// ═══════════════════════════════════════════════════════

import type { ThemeSetting } from '../hooks/useTheme';
import { VIEWS, type View } from '../hooks/useRoute';

const LABELS: Record<View, string> = {
  desk: 'Desk',
  ask: 'Ask',
  findings: 'Findings',
  watchlist: 'Watchlist',
  system: 'System',
};

export function Header({
  view,
  navigate,
  pendingCount,
  theme,
  setTheme,
}: {
  view: View;
  navigate: (view: View) => void;
  pendingCount: number;
  theme: ThemeSetting;
  setTheme: (theme: ThemeSetting) => void;
}) {
  return (
    <header
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 20,
        display: 'flex',
        flexWrap: 'wrap',
        rowGap: 10,
        alignItems: 'center',
        gap: 32,
        padding: '15px 40px',
        background: 'color-mix(in srgb, var(--color-bg) 90%, transparent)',
        backdropFilter: 'blur(14px)',
        borderBottom: '1px solid var(--color-divider)',
      }}
    >
      <span
        style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 600,
          fontSize: 19,
          letterSpacing: '0.01em',
          flex: 'none',
        }}
      >
        FinSight
      </span>

      <nav style={{ display: 'flex', gap: 2, flex: 'none' }} aria-label="Views">
        {VIEWS.map((key) => {
          const active = view === key;
          return (
            <button
              key={key}
              type="button"
              onClick={() => navigate(key)}
              aria-current={active ? 'page' : undefined}
              style={{
                fontFamily: 'var(--font-heading)',
                fontWeight: 600,
                fontSize: 14,
                letterSpacing: '0.02em',
                background: 'transparent',
                border: '1px solid transparent',
                borderBottomColor: active ? 'var(--color-accent)' : 'transparent',
                padding: '6px 12px',
                cursor: 'pointer',
                color: active ? 'var(--color-text)' : 'var(--ink-45)',
              }}
            >
              {LABELS[key]}
              {key === 'desk' && pendingCount > 0 ? (
                <span style={{ fontSize: 11, opacity: 0.75, marginLeft: 6 }}>{pendingCount}</span>
              ) : null}
            </button>
          );
        })}
      </nav>

      <div className="seg" style={{ flex: 'none', marginLeft: 'auto' }} role="radiogroup" aria-label="Theme">
        {(['light', 'auto', 'dark'] as ThemeSetting[]).map((option) => (
          <label key={option} className="seg-opt" style={{ padding: '5px 9px', fontSize: 11 }}>
            <input
              type="radio"
              name="fs-theme"
              checked={theme === option}
              onChange={() => setTheme(option)}
            />
            {option[0].toUpperCase() + option.slice(1)}
          </label>
        ))}
      </div>
    </header>
  );
}
