// ═══════════════════════════════════════════════════════
// FinSight web — application shell
// ═══════════════════════════════════════════════════════
//
// ══ WHY THE PENDING QUEUE IS LOADED HERE AND NOT IN Desk ══
//   The count of decisions waiting is on the header badge, which is visible
//   from every view. Loading it inside Desk would mean the badge is only
//   correct on the screen that already tells you. It polls, because the whole
//   premise of the watching half is that things happen while nobody is
//   looking — and the scheduler can pause a cycle with the app open.
// ═══════════════════════════════════════════════════════

import { Header } from './components/Header';
import { useResource } from './hooks/useResource';
import { useRoute } from './hooks/useRoute';
import { useTheme } from './hooks/useTheme';
import { Ask } from './views/Ask';
import { Desk, loadPending } from './views/Desk';
import { Findings } from './views/Findings';
import { System } from './views/System';
import { Watchlist } from './views/Watchlist';

export function App() {
  const { view, navigate } = useRoute();
  const { theme, setTheme, vizColors } = useTheme();

  const pending = useResource(() => loadPending(), [], { pollMs: 45_000 });
  const pendingCount = (pending.data ?? []).reduce((total, entry) => total + entry.alerts.length, 0);

  return (
    // A column with the main region flexed to fill: the quiet Desk is short,
    // and a footer that stops halfway down an empty screen reads as a page
    // that failed to finish loading rather than one with nothing to say.
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--color-bg)',
        color: 'var(--color-text)',
      }}
    >
      <Header view={view} navigate={navigate} pendingCount={pendingCount} theme={theme} setTheme={setTheme} />

      <div style={{ flex: '1 0 auto' }}>
        {view === 'desk' ? (
          <Desk
            pending={pending.data}
            pendingError={pending.error}
            reloadPending={pending.reload}
            vizColors={vizColors}
            navigate={navigate}
          />
        ) : null}
        {view === 'ask' ? <Ask /> : null}
        {view === 'findings' ? <Findings onCycleFinished={pending.reload} /> : null}
        {view === 'watchlist' ? <Watchlist /> : null}
        {view === 'system' ? <System /> : null}
      </div>

      <footer
        style={{
          flex: 'none',
          borderTop: '1px solid var(--color-divider)',
          padding: '22px 40px',
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 12,
          fontSize: 12,
          color: 'var(--ink-42)',
        }}
      >
        <span>
          Severity reads [!!] HIGH, [~] MED, [.] LOW, so it survives greyscale. Colour only repeats the marker.
        </span>
        <span>No live market data — every source is a delayed free tier.</span>
      </footer>
    </div>
  );
}
