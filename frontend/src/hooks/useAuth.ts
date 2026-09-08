// ═══════════════════════════════════════════════════════
// FinSight web — Google sign-in
// ═══════════════════════════════════════════════════════
//
// Asks the API whether it wants a sign-in, loads Google Identity Services if
// so, and holds the resulting ID token for src/api/client.ts to send.
//
// ══ WHY THE CLIENT ID IS FETCHED, NOT IMPORTED ══
//   Vite inlines env vars at BUILD time. A VITE_GOOGLE_CLIENT_ID would tie
//   every built image to one Google project, which is the property
//   frontend/Dockerfile goes out of its way to avoid for VITE_API_URL. So the
//   ID arrives from GET /auth/config at runtime and one image runs anywhere.
//
// ══ WHY renderButton AND NOT One Tap ══
//   google.accounts.id.prompt() is entangled with third-party-cookie
//   deprecation and the FedCM migration, and degrades differently across
//   browsers depending on both. The rendered button is an explicit user
//   gesture opening a first-party popup — the one flow that behaves the same
//   everywhere. A dashboard nobody can sign into is worse than one extra
//   click.
//
// ══ THE TOKEN IS NOT PERSISTED ══
//   It lives in memory only; see the note in src/api/client.ts. A reload
//   costs a re-acquire, which is silent whenever Google still has a session
//   for the account (auto_select), and a visible button when it does not.
// ═══════════════════════════════════════════════════════

import { useCallback, useEffect, useRef, useState } from 'react';
import { authConfig, setAuthToken, setUnauthorizedHandler } from '../api/client';

const GSI_SRC = 'https://accounts.google.com/gsi/client';

/** Minimal shape of the bits of GIS we use — the library ships no types. */
interface GoogleIdentity {
  accounts: {
    id: {
      initialize(config: {
        client_id: string;
        callback: (response: { credential?: string }) => void;
        auto_select?: boolean;
        cancel_on_tap_outside?: boolean;
      }): void;
      renderButton(parent: HTMLElement, options: Record<string, unknown>): void;
      disableAutoSelect(): void;
    };
  };
}

declare global {
  interface Window {
    google?: GoogleIdentity;
  }
}

export type AuthStatus =
  /** Still asking the API whether it wants a sign-in. */
  | 'loading'
  /** Auth is off, or a token is held. Either way: show the dashboard. */
  | 'ready'
  /** Auth is on and there is no token. Show the gate. */
  | 'signed-out'
  /** The API could not be reached, or GIS would not load. */
  | 'error';

export interface Auth {
  status: AuthStatus;
  /** The signed-in address, when known. Empty when auth is off. */
  email: string;
  /** Set when status is 'error', or when a sign-in was refused. */
  message: string;
  /** Attach the Google button to a container element. No-op when auth is off. */
  mountButton: (element: HTMLElement | null) => void;
  signOut: () => void;
}

/** Load the GIS script once, resolving when window.google is usable. */
function loadGsi(): Promise<GoogleIdentity> {
  if (window.google?.accounts?.id) return Promise.resolve(window.google);

  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GSI_SRC}"]`);
    const script = existing ?? document.createElement('script');

    const done = () => {
      if (window.google?.accounts?.id) resolve(window.google);
      else reject(new Error('Google Identity Services loaded but exposed no accounts.id'));
    };

    script.addEventListener('load', done, { once: true });
    script.addEventListener('error', () => reject(new Error('Could not load Google Identity Services')), {
      once: true,
    });

    if (!existing) {
      script.src = GSI_SRC;
      script.async = true;
      script.defer = true;
      document.head.appendChild(script);
    }
  });
}

/** Read the email out of an ID token's payload, for display only. */
function emailFromToken(token: string): string {
  try {
    const payload = token.split('.')[1];
    if (!payload) return '';
    // base64url -> base64. The API verifies this token properly; this is a
    // label for the header and is never trusted for anything.
    const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    return String((JSON.parse(json) as { email?: string }).email ?? '');
  } catch {
    return '';
  }
}

export function useAuth(): Auth {
  const [status, setStatus] = useState<AuthStatus>('loading');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const identity = useRef<GoogleIdentity | null>(null);
  const pendingMount = useRef<HTMLElement | null>(null);

  const renderInto = useCallback((element: HTMLElement) => {
    if (!identity.current) return;
    element.replaceChildren();
    identity.current.accounts.id.renderButton(element, {
      type: 'standard',
      theme: 'outline',
      size: 'large',
      text: 'signin_with',
      shape: 'rectangular',
    });
  }, []);

  useEffect(() => {
    let cancelled = false;

    // A 401 from any later call means the token expired or was revoked. Clear
    // the identity and fall back to the gate; client.ts has already dropped
    // the token by the time this runs.
    //
    // Neither 403 nor 402 comes through here. 403 means the route is reserved
    // for the unlimited tier and 402 means the free queries are spent — a new
    // token fixes neither, so re-prompting would loop against a wall. The 402
    // is handled where it can be acted on, in Ask.tsx.
    setUnauthorizedHandler(() => {
      if (cancelled) return;
      setEmail('');
      setMessage('Your session expired. Sign in again to continue.');
      setStatus('signed-out');
    });

    void (async () => {
      let config;
      try {
        config = await authConfig();
      } catch (exc) {
        if (cancelled) return;
        setMessage(exc instanceof Error ? exc.message : String(exc));
        setStatus('error');
        return;
      }
      if (cancelled) return;

      // Auth off: no Google, no gate, no setup needed for `npm run dev`.
      if (!config.enabled) {
        setStatus('ready');
        return;
      }

      try {
        const google = await loadGsi();
        if (cancelled) return;

        google.accounts.id.initialize({
          client_id: config.client_id,
          auto_select: true,
          cancel_on_tap_outside: false,
          callback: (response) => {
            if (cancelled) return;
            if (!response.credential) {
              setMessage('Google returned no credential. Try again.');
              setStatus('signed-out');
              return;
            }
            setAuthToken(response.credential);
            setEmail(emailFromToken(response.credential));
            setMessage('');
            setStatus('ready');
          },
        });

        identity.current = google;
        setStatus('signed-out');

        // The gate may have mounted its container before GIS finished loading.
        if (pendingMount.current) renderInto(pendingMount.current);
      } catch (exc) {
        if (cancelled) return;
        setMessage(exc instanceof Error ? exc.message : String(exc));
        setStatus('error');
      }
    })();

    return () => {
      cancelled = true;
      setUnauthorizedHandler(null);
    };
  }, [renderInto]);

  const mountButton = useCallback(
    (element: HTMLElement | null) => {
      pendingMount.current = element;
      if (element && identity.current) renderInto(element);
    },
    [renderInto],
  );

  const signOut = useCallback(() => {
    setAuthToken(null);
    // Without this, auto_select silently signs the same account straight back
    // in and the button appears to do nothing.
    identity.current?.accounts.id.disableAutoSelect();
    setEmail('');
    setMessage('');
    setStatus('signed-out');
  }, []);

  return { status, email, message, mountButton, signOut };
}
