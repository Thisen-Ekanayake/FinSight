// ═══════════════════════════════════════════════════════
// FinSight web — routing
// ═══════════════════════════════════════════════════════
//
// Five views, no nesting, no params. Real paths (/findings) rather than a
// fragment (#findings), and still no router dependency — five flat routes do
// not need one.
//
// ══ WHAT MAKES THE PATHS WORK ══
//   The app ships as static files, so a hard refresh on /findings asks the
//   server for a file that does not exist. Both servers are already set up to
//   answer it with index.html: nginx.conf's `try_files $uri $uri/ /index.html`
//   in the container, and Vite's own SPA fallback under `npm run dev`. Break
//   either and every deep link 404s while in-app navigation still works —
//   which is why nginx.conf calls that line out as load-bearing.
//
// ══ FRAGMENTS ARE STILL FRAGMENTS ══
//   The anchor targets in the Desk hero (href="#queue") are same-page jumps,
//   not routes; on /desk the browser resolves them to /desk#queue and scrolls,
//   which is exactly what they always meant. `fromPath` matching only the five
//   view slugs keeps them out of routing, and an unknown path falls back to
//   the default view rather than rendering nothing.
// ═══════════════════════════════════════════════════════

import { useCallback, useEffect, useState } from 'react';

export const VIEWS = ['desk', 'ask', 'findings', 'watchlist', 'system'] as const;
export type View = (typeof VIEWS)[number];

/** A view slug, or null when this is not one of the five routes. */
function fromPath(path: string): View | null {
  const slug = path.replace(/^\/+|\/+$/g, '').split('?')[0];
  return (VIEWS as readonly string[]).includes(slug) ? (slug as View) : null;
}

/**
 * A view from a #findings-style URL left over from the hash-routing build.
 *
 * Those links are in the wild — bookmarked, and pasted wherever the deployed
 * app has been shared — so they are honoured once on load and rewritten to the
 * path form, rather than silently dumping the visitor on the Desk.
 */
function fromLegacyHash(): View | null {
  return fromPath(window.location.hash.replace(/^#\/?/, ''));
}

export function useRoute(): { view: View; navigate: (view: View) => void } {
  const [view, setView] = useState<View>(() => fromLegacyHash() ?? fromPath(window.location.pathname) ?? 'desk');

  useEffect(() => {
    const legacy = fromLegacyHash();
    if (legacy) window.history.replaceState(null, '', `/${legacy}`);
  }, []);

  useEffect(() => {
    // popstate, not hashchange: the back button is now the only thing that
    // changes the location without going through `navigate`.
    const onPopState = () => setView(fromPath(window.location.pathname) ?? 'desk');
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const navigate = useCallback((next: View) => {
    // pushState fires no event, so the state has to be set here as well.
    // Skipped when the URL already says this — re-clicking the current view in
    // the header would otherwise stack history entries the back button has to
    // chew through. Comparing the fragment too means a jump to #queue is
    // cleared by pressing Desk again.
    if (`${window.location.pathname}${window.location.hash}` !== `/${next}`) {
      window.history.pushState(null, '', `/${next}`);
    }
    setView(next);
    // Landing halfway down the previous view's scroll position reads as a
    // broken page rather than a new one.
    window.scrollTo({ top: 0 });
  }, []);

  return { view, navigate };
}
