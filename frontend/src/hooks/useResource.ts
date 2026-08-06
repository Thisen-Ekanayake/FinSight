// ═══════════════════════════════════════════════════════
// FinSight web — resource loading
// ═══════════════════════════════════════════════════════
//
// One hook for "fetch this, show an error if it fails, optionally keep it
// fresh". The dashboard has no client-side store and holds no state of its
// own (design brief) — every screen is a read of the backend plus, at most,
// a form.
//
// ══ WHY loading IS FALSE ON A REFRESH ══
//   A poll that flipped `loading` back to true every interval would blank the
//   screen on a timer. `loading` is true only until the FIRST result; after
//   that a refresh swaps `data` underneath whatever is already on screen and
//   a failed refresh keeps the last good value while surfacing the error.
//   Losing a rendered answer because a background poll timed out would be a
//   worse outcome than showing one that is a few seconds stale.
// ═══════════════════════════════════════════════════════

import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from '../api/client';

export interface Resource<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  /** Re-run the fetch now. Safe to call from an event handler. */
  reload: () => void;
}

export function useResource<T>(
  fetcher: () => Promise<T>,
  deps: readonly unknown[] = [],
  opts: { pollMs?: number; enabled?: boolean } = {},
): Resource<T> {
  const { pollMs, enabled = true } = opts;

  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [nonce, setNonce] = useState(0);

  // The fetcher is a fresh closure on every render; keeping it in a ref means
  // the effect below depends on `deps` alone rather than re-firing constantly.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }

    let live = true;
    let timer: number | undefined;

    const run = async () => {
      try {
        const result = await fetcherRef.current();
        if (!live) return;
        setData(result);
        setError(null);
      } catch (exc) {
        if (!live) return;
        setError(exc instanceof ApiError ? exc.message : String(exc));
      } finally {
        if (live) setLoading(false);
        // Chained rather than setInterval: a slow response must not stack up
        // a queue of overlapping requests behind it.
        if (live && pollMs) timer = window.setTimeout(run, pollMs);
      }
    };

    void run();
    return () => {
      live = false;
      if (timer) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce, pollMs, enabled]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  return { data, error, loading, reload };
}
