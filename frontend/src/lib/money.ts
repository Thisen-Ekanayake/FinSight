// ═══════════════════════════════════════════════════════
// FinSight web — money formatting
// ═══════════════════════════════════════════════════════
// Signs are always explicit on deltas (+/−), so a gain and a loss differ in
// text and not only in colour.
// ═══════════════════════════════════════════════════════

const usd = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
const usd0 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

export function money(value: number | null | undefined, whole = false): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—';
  return (whole ? usd0 : usd).format(value);
}

export function signedMoney(value: number, whole = false): string {
  const body = money(Math.abs(value), whole);
  return value > 0 ? `+${body}` : value < 0 ? `−${body}` : body;
}

export function pct(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '—';
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function signedPct(fraction: number, digits = 1): string {
  const body = pct(Math.abs(fraction), digits);
  return fraction > 0 ? `+${body}` : fraction < 0 ? `−${body}` : body;
}

export function thisMonth(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
}

export function shiftMonth(month: string, delta: number): string {
  const [y, m] = month.split('-').map(Number);
  const d = new Date(y, m - 1 + delta, 1);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`;
}

export function monthLabel(month: string): string {
  const [y, m] = month.split('-').map(Number);
  return new Date(y, m - 1, 1).toLocaleDateString(undefined, { month: 'long', year: 'numeric' });
}

export function today(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** Tone for "how much of a limit is used": calm, nearly there, over. */
export function usageTone(used: number, limit: number): string {
  if (limit <= 0) return 'var(--color-accent)';
  const r = used / limit;
  return r > 1 ? 'var(--neg)' : r > 0.85 ? 'var(--warn)' : 'var(--pos)';
}
