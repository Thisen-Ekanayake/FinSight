// ═══════════════════════════════════════════════════════
// FinSight web — application shell
// ═══════════════════════════════════════════════════════
//
// ══ WHY THE PENDING QUEUE IS LOADED HERE AND NOT IN Desk ══
//   The count of decisions waiting is on the sidebar badge, which is visible
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
import { Sidebar } from './components/Sidebar';
import { Loading } from './components/primitives';
import { ToastProvider } from './components/ui';
import { useAuth } from './hooks/useAuth';
import { useResource } from './hooks/useResource';
import { useRoute } from './hooks/useRoute';
import { useTheme } from './hooks/useTheme';
import { Agents } from './views/Agents';
import { Ask } from './views/Ask';
import { Budget } from './views/Budget';
import { Desk, loadPending } from './views/Desk';
import { Findings } from './views/Findings';
import { Overview } from './views/Overview';
import { Portfolio } from './views/Portfolio';
import { SignIn } from './views/SignIn';
import { System } from './views/System';
import { Watchlist } from './views/Watchlist';

export function App() {
  const { view, navigate } = useRoute();
  const { theme, setTheme, vizColors } = useTheme();
  const auth = useAuth();
  const authed = auth.status === 'ready';

  // `enabled` matters more than it looks: this poll runs at the shell level so
  // the sidebar badge is right on every view, which means an unauthenticated
  // session would otherwise fire a 401 every 45 seconds, forever.
  const pending = useResource(() => loadPending(), [authed], { pollMs: 45_000, enabled: authed });
  const pendingCount = (pending.data ?? []).reduce((total, entry) => total + entry.alerts.length, 0);

  // Order proposals waiting on a decision — the Portfolio badge. Same reasoning
  // as the alert badge: it has to be right from every screen.
  const proposals = useResource(() => api.listProposals('PENDING'), [authed, view], { pollMs: 60_000, enabled: authed });
  const proposalCount = proposals.data?.length ?? 0;

  // A free-tier account is refused the monitor routes, so `pending` above will
  // 403 for them — useResource keeps that in `error` and the badge simply
  // stays at zero, which is the right outcome: there is nothing they can act on.
  const quota = useResource(() => api.quota(), [authed], { enabled: authed });

  // Deliberately before the shell: rendering the sidebar and a view first
  // would fire every view's own loader against a guarded API.
  if (auth.status === 'loading') {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--color-bg)', color: 'var(--color-text)', padding: 40 }}>
        <Loading label="Checking sign-in…" />
      </div>
    );
  }

  if (!authed) return <SignIn auth={auth} />;

  return (
    <ToastProvider>
      <div className="shell">
        <Sidebar
          view={view}
          navigate={navigate}
          pendingCount={pendingCount}
          proposalCount={proposalCount}
          theme={theme}
          setTheme={setTheme}
          email={auth.email}
          onSignOut={auth.signOut}
        />

        <main className="main">
          <div style={{ flex: '1 0 auto' }} key={view}>
            {view === 'overview' ? <Overview navigate={navigate} pendingCount={pendingCount} email={auth.email} /> : null}
            {view === 'budget' ? <Budget /> : null}
            {view === 'portfolio' ? <Portfolio navigate={navigate} /> : null}
            {view === 'agents' ? <Agents navigate={navigate} /> : null}
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
              padding: '18px 32px',
              display: 'flex',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 12,
              fontSize: 12,
              color: 'var(--ink-45)',
            }}
          >
            <span>Not financial advice. Agents learn from past prices; orders go only to a paper account, after your approval.</span>
            <span style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <span>Market data is delayed, from free sources.</span>
              {/* The URL comes from the API rather than a constant here, for the
                  same reason the Google client ID does: one bundle, any deployment. */}
              {quota.data?.contact_url ? (
                <a href={quota.data.contact_url} target="_blank" rel="noreferrer noopener" style={{ color: 'var(--ink-45)' }}>
                  Source
                </a>
              ) : null}
            </span>
          </footer>
        </main>
      </div>
    </ToastProvider>
  );
}
