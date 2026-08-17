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
//
// ══ WHY THE QUOTA IS LOADED HERE TOO ══
//   Same reasoning, different reason to care: the remaining count belongs
//   next to the question box, but it is the shell that knows when a sign-in
//   has just happened. It does NOT poll — nothing spends a query except this
//   browser, so it reloads on demand after Ask finishes one.
// ═══════════════════════════════════════════════════════

import * as api from './api/client';
import { Header } from './components/Header';
import { Loading } from './components/primitives';
import { useAuth } from './hooks/useAuth';
import { useResource } from './hooks/useResource';
import { useRoute } from './hooks/useRoute';
import { useTheme } from './hooks/useTheme';
import { Ask } from './views/Ask';
import { Desk, loadPending } from './views/Desk';
import { Findings } from './views/Findings';
import { SignIn } from './views/SignIn';
import { System } from './views/System';
import { Watchlist } from './views/Watchlist';

export function App() {
  const { view, navigate } = useRoute();
  const { theme, setTheme, vizColors } = useTheme();
  const auth = useAuth();
  const authed = auth.status === 'ready';

  // `enabled` matters more than it looks: this poll runs at the shell level so
  // the header badge is right on every view, which means an unauthenticated
  // session would otherwise fire a 401 every 45 seconds, forever.
  const pending = useResource(() => loadPending(), [authed], { pollMs: 45_000, enabled: authed });
  const pendingCount = (pending.data ?? []).reduce((total, entry) => total + entry.alerts.length, 0);

  // A free-tier account is refused the monitor routes, so `pending` above will
  // 403 for them — useResource keeps that in `error` and the badge simply
  // stays at zero, which is the right outcome: there is nothing they can act on.
  const quota = useResource(() => api.quota(), [authed], { enabled: authed });

  // Deliberately before the shell: rendering Header and a view first would
  // fire every view's own loader against a guarded API.
  if (auth.status === 'loading') {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--color-bg)', color: 'var(--color-text)' }}>
        <Loading label="Checking sign-in…" />
      </div>
    );
  }

  if (!authed) return <SignIn auth={auth} />;

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
      <Header
        view={view}
        navigate={navigate}
        pendingCount={pendingCount}
        theme={theme}
        setTheme={setTheme}
        email={auth.email}
        onSignOut={auth.signOut}
      />

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
        {view === 'ask' ? <Ask quota={quota.data} onSpend={quota.reload} /> : null}
        {view === 'findings' ? <Findings onCycleFinished={pending.reload} vizColors={vizColors} /> : null}
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
        <span style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <span>No live market data — every source is a delayed free tier.</span>
          {/* The URL comes from the API rather than a constant here, for the
              same reason the Google client ID does: one bundle, any deployment. */}
          {quota.data?.contact_url ? (
            <a
              href={quota.data.contact_url}
              target="_blank"
              rel="noreferrer noopener"
              style={{ color: 'var(--ink-42)' }}
            >
              Source
            </a>
          ) : null}
        </span>
      </footer>
    </div>
  );
}
