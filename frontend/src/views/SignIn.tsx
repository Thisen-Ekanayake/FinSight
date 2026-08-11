// ═══════════════════════════════════════════════════════
// FinSight web — sign-in gate
// ═══════════════════════════════════════════════════════
//
// Shown instead of the dashboard while auth is on and no token is held.
//
// ══ WHY IT EXPLAINS ITSELF ══
//   This is a single-operator tool with an allowlist, so most people who ever
//   reach this screen cannot get past it. Saying so here — rather than letting
//   them sign in and meet an unexplained 403 — is the difference between a
//   closed door and a broken one.
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
          Every question costs real model quota, and approving a HIGH-severity alert dispatches it. Both are
          recorded against the account that did them, so this deployment serves a fixed list of addresses
          rather than anyone with the link.
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
          Not on the list? Signing in will succeed and still be refused — the account is verified, it is simply
          not permitted here.
        </p>
      </Blueprint>
    </div>
  );
}
