// ═══════════════════════════════════════════════════════
// FinSight web — sign-in gate
// ═══════════════════════════════════════════════════════
//
// Shown instead of the dashboard while auth is on and no token is held.
//
// ══ WHY IT EXPLAINS ITSELF ══
//   Anyone with a Google account can get past this screen now, and gets a
//   small fixed number of questions once they do. Saying so here — rather
//   than letting someone spend their allowance discovering it existed — is
//   what makes the limit a stated offer instead of an ambush.
//
//   It said the opposite until the free tier landed: the allowlist used to
//   decide admission, so this copy warned that signing in would probably
//   still be refused. That is no longer true of any Google account.
// ═══════════════════════════════════════════════════════

import { useCallback } from 'react';
import { Blueprint, ErrorNote, Kicker } from '../components/primitives';
import type { Auth } from '../hooks/useAuth';

export function SignIn({ auth }: { auth: Auth }) {
  // A callback ref rather than useEffect: the container element is what GIS
  // needs, and this fires exactly when it exists.
  const attach = useCallback(
    (element: HTMLDivElement | null) => {
      auth.mountButton(element);
    },
    [auth],
  );

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--color-bg)',
        color: 'var(--color-text)',
        padding: 24,
      }}
    >
      <Blueprint style={{ maxWidth: 460, width: '100%', padding: '36px 34px' }}>
        <Kicker>FinSight</Kicker>

        <h1 style={{ fontSize: 21, lineHeight: 1.3, margin: '0 0 14px', fontWeight: 600 }}>
          Multi-agent financial research, with enforced citations
        </h1>

        <p className="mono" style={{ fontSize: 12.5, lineHeight: 1.7, color: 'var(--ink-70)', margin: '0 0 22px' }}>
          Sign in with any Google account to try it. Every question runs a full multi-agent research pass
          against live sources, so a new account gets a small fixed number of them — the dashboard, the
          findings and the audit trail stay open either way.
        </p>

        {auth.status === 'error' ? (
          <ErrorNote error={auth.message || 'Sign-in is unavailable.'} />
        ) : (
          <ErrorNote error={auth.message || null} />
        )}

        {/* GIS renders its own button in here. Nothing of ours goes inside:
            the library replaces the contents on every render. */}
        <div ref={attach} style={{ display: 'flex', justifyContent: 'flex-start', minHeight: 44 }} />

        <p className="mono" style={{ fontSize: 11.5, lineHeight: 1.65, color: 'var(--ink-42)', margin: '22px 0 0' }}>
          Google tells this deployment your email address and nothing else. Running the monitoring side, which
          dispatches alerts, stays with the operator.
        </p>
      </Blueprint>
    </div>
  );
}
