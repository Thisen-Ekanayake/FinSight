// ═══════════════════════════════════════════════════════
// FinSight web — routing
// ═══════════════════════════════════════════════════════
//
// Five views, no nesting, no params. A hash route rather than a router
// dependency, and hash rather than history: the app ships as static files
// behind nginx and a deep link to /findings on a hard refresh would 404
// unless every path were rewritten to index.html. #findings cannot.
//
// The anchor targets in the Desk hero (href="#queue") are same-page jumps,
// not routes, so `parse` treats an unknown hash as the default view rather
// than rendering nothing.
// ═══════════════════════════════════════════════════════

import { useCallback, useEffect, useState } from 'react';

export const VIEWS = ['desk', 'ask', 'findings', 'watchlist', 'system'] as const;
export type View = (typeof VIEWS)[number];

function parse(): View {
  const hash = window.location.hash.replace(/^#\/?/, '').split('?')[0];
  return (VIEWS as readonly string[]).includes(hash) ? (hash as View) : 'desk';
}

export function useRoute(): { view: View; navigate: (view: View) => void } {
  const [view, setView] = useState<View>(parse);

  useEffect(() => {
    const onHashChange = () => setView(parse());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = useCallback((next: View) => {
    window.location.hash = next;
    // Landing halfway down the previous view's scroll position reads as a
    // broken page rather than a new one.
    window.scrollTo({ top: 0 });
  }, []);

  return { view, navigate };
}
