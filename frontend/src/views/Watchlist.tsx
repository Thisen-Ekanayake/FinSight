// ═══════════════════════════════════════════════════════
// FinSight web — Watchlist
// ═══════════════════════════════════════════════════════
//
// Purpose : What the machine watches, and whether each name has been read yet.
//
// ══ COLD IS A STATE, NOT AN ERROR ══
//   A brand-new ticker has no history indexed, so the dedup engine has
//   nothing to compare a candidate against and the first ordinary cycle would
//   report everything it saw as new. `warmed_up` is the flag for that, and a
//   cold ticker is shown as cold rather than as a problem — the fix is a
//   warm-up run, which is a normal part of adding a name.
//
// ══ WHY REMOVAL ASKS ══
//   DELETE /watchlist/{ticker} is a soft delete, so nothing is lost. But the
//   row is the only place the ticker appears, and a mis-click that silently
//   stops monitoring a company is exactly the failure this product cannot
//   afford. One confirm, inline, no dialog.
// ═══════════════════════════════════════════════════════

import { useState } from 'react';
import * as api from '../api/client';
import type { WatchItem } from '../api/types';
import { useResource } from '../hooks/useResource';
import { Empty, ErrorNote, Loading, PageHead } from '../components/primitives';
import { since, stamp } from '../lib/format';

export function Watchlist() {
  const watchlist = useResource(() => api.getWatchlist(), []);
  const [ticker, setTicker] = useState('');
  const [company, setCompany] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  const add = async (event: React.FormEvent) => {
    event.preventDefault();
    const symbol = ticker.trim().toUpperCase();
    if (!symbol || busy) return;

    setBusy(true);
    setError(null);
    try {
      await api.addTicker(symbol, company.trim());
      setTicker('');
      setCompany('');
      watchlist.reload();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (symbol: string) => {
    setBusy(true);
    setError(null);
    try {
      await api.removeTicker(symbol);
      setConfirming(null);
      watchlist.reload();
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  };

  const items = watchlist.data ?? [];

  return (
    <main style={{ maxWidth: 760, margin: '0 auto', padding: '72px 40px 120px' }}>
      <PageHead title="What it watches">
        Cold tickers are skipped by the dedup engine until a warm-up run has read their history — a new name behaves
        differently until then.
      </PageHead>

      <form onSubmit={add} style={{ display: 'flex', gap: 10, alignItems: 'flex-end', marginBottom: 34, flexWrap: 'wrap' }}>
        <div className="field" style={{ flex: '0 0 110px' }}>
          <label htmlFor="fs-ticker">Ticker</label>
          <input
            id="fs-ticker"
            className="input"
            placeholder="AMZN"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            maxLength={16}
          />
        </div>
        <div className="field" style={{ flex: '1 1 220px' }}>
          <label htmlFor="fs-company">Company — looked up from EDGAR if blank</label>
          <input
            id="fs-company"
            className="input"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            maxLength={128}
          />
        </div>
        <button className="btn btn-secondary" type="submit" disabled={busy || !ticker.trim()} style={{ flex: 'none' }}>
          Add
        </button>
      </form>

      <ErrorNote error={error ?? watchlist.error} />

      {watchlist.loading ? <Loading /> : null}

      {!watchlist.loading && items.length === 0 ? (
        <Empty headline="Nothing is being watched.">
          Add a ticker above and the monitoring half has something to do. Five to twenty names is the shape this is
          built for.
        </Empty>
      ) : null}

      {items.map((item) => (
        <Row
          key={item.ticker}
          item={item}
          confirming={confirming === item.ticker}
          busy={busy}
          onAskRemove={() => setConfirming(item.ticker)}
          onCancel={() => setConfirming(null)}
          onConfirm={() => remove(item.ticker)}
        />
      ))}

      {items.length > 0 ? (
        <p style={{ fontSize: 13, color: 'var(--ink-52)', marginTop: 22, maxWidth: '60ch', textWrap: 'pretty' }}>
          Each name carries a per-monitor watermark — the last time filings, news, prices, and macro were checked for
          it. A run only looks at what has happened since.
        </p>
      ) : null}
    </main>
  );
}

function Row({
  item,
  confirming,
  busy,
  onAskRemove,
  onCancel,
  onConfirm,
}: {
  item: WatchItem;
  confirming: boolean;
  busy: boolean;
  onAskRemove: () => void;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  // The most recent watermark across every monitor: "when was this last read".
  const watermarks = Object.values(item.last_checked ?? {}).filter(Boolean);
  const lastRead = watermarks.length > 0 ? watermarks.sort().slice(-1)[0] : null;

  return (
    <div style={{ display: 'flex', gap: 16, alignItems: 'baseline', padding: '15px 2px', borderTop: '1px solid var(--color-divider)', flexWrap: 'wrap' }}>
      <span
        style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 600,
          fontSize: 16,
          letterSpacing: '0.06em',
          flex: '0 0 66px',
          color: item.warmed_up ? 'var(--color-text)' : 'var(--ink-52)',
        }}
      >
        {item.ticker}
      </span>
      <span style={{ flex: '1 1 160px', minWidth: 0, fontSize: 14.5 }}>{item.company_name || '—'}</span>

      <span className="mono" style={{ fontSize: 11.5, flex: 'none', color: 'var(--ink-52)' }} title={lastRead ? stamp(lastRead) : undefined}>
        {item.warmed_up
          ? lastRead
            ? `read ${since(lastRead)} ago`
            : 'warmed up'
          : lastRead
            ? `cold — read ${since(lastRead)} ago`
            : 'cold — never read'}
      </span>

      {confirming ? (
        <span style={{ display: 'flex', gap: 10, alignItems: 'center', flex: 'none' }}>
          <span style={{ fontSize: 12.5, color: 'var(--ink-62)' }}>Stop watching?</span>
          <button className="btn btn-secondary" type="button" onClick={onConfirm} disabled={busy} style={{ padding: '3px 9px', fontSize: 12 }}>
            Yes
          </button>
          <button className="btn btn-ghost" type="button" onClick={onCancel} style={{ padding: '3px 6px', fontSize: 12 }}>
            Keep
          </button>
        </span>
      ) : (
        <button
          className="btn btn-ghost"
          type="button"
          onClick={onAskRemove}
          aria-label={`Stop watching ${item.ticker}`}
          style={{ padding: '3px 6px', fontSize: 12, flex: 'none' }}
        >
          Remove
        </button>
      )}
    </div>
  );
}
