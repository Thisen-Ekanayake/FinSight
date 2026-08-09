// ═══════════════════════════════════════════════════════
// FinSight web — formatting
// ═══════════════════════════════════════════════════════
//
// The vocabulary the interface repeats: severity markers, elapsed time, and
// the handful of number shapes that appear in more than one view.
// ═══════════════════════════════════════════════════════

import type { Severity } from '../api/types';

/**
 * Plain-text severity markers, matching src/ui/components.py.
 *
 * ══ WHY A MARKER AND NOT JUST A COLOUR ══
 *   Severity has to survive greyscale and colour-vision deficiency. This is a
 *   financial alert stream whose failure mode is someone missing a HIGH, so
 *   colour may only ever REPEAT what the marker already says. Every place
 *   that tints a severity also renders this string.
 */
const MARKERS: Record<string, string> = { HIGH: '[!!]', MED: '[~]', LOW: '[.]' };

export function severityMark(severity: string): string {
  return MARKERS[severity?.toUpperCase()] ?? '[?]';
}

/** The CSS colour a severity is allowed to be tinted. */
export function severityInk(severity: string): string {
  const s = severity?.toUpperCase();
  if (s === 'HIGH') return 'var(--alert-ink)';
  if (s === 'MED') return 'var(--color-accent)';
  return 'var(--ink-52)';
}

export function isSeverity(value: string): value is Severity {
  return value === 'HIGH' || value === 'MED' || value === 'LOW';
}

/** "14:30" in the viewer's own zone. Timestamps arrive from the API as UTC ISO. */
export function clock(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.valueOf())
    ? '—'
    : d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', hour12: false });
}

/** "6 Aug, 14:30" — used where a time alone would be ambiguous across days. */
export function stamp(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.valueOf())) return '—';
  return d.toLocaleString(undefined, {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

/**
 * "held 2 h 14 min" — how long something has been waiting.
 *
 * A paused run survives a restart and can sit for days, so this deliberately
 * keeps counting up in days rather than collapsing to "a while ago". Stale
 * items are normal here, not a bug, and the age is the useful part.
 */
export function since(iso: string | null | undefined): string {
  if (!iso) return '—';
  const then = new Date(iso).valueOf();
  if (Number.isNaN(then)) return '—';
  const mins = Math.max(0, Math.floor((Date.now() - then) / 60000));
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} h ${String(mins % 60).padStart(2, '0')} min`;
  const days = Math.floor(hours / 24);
  return `${days} d ${hours % 24} h`;
}

/** Milliseconds as seconds, one decimal: "38.4s". */
export function seconds(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return '—';
  return `${(ms / 1000).toFixed(1)}s`;
}

/** 0.923 -> "92%". */
export function percent(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || Number.isNaN(fraction)) return '—';
  return `${(fraction * 100).toFixed(digits)}%`;
}

/**
 * A financial magnitude at reading length: 416161000000 -> "416.2B".
 *
 * Thousands, never 1024 — these are dollars and share counts, not bytes.
 * Trailing zeros are dropped only after a decimal point, so an axis tick of
 * 100 stays "100" rather than collapsing to "1".
 */
export function compact(value: number): string {
  if (!Number.isFinite(value)) return '—';

  const abs = Math.abs(value);
  const [div, suffix] =
    abs >= 1e12
      ? ([1e12, 'T'] as const)
      : abs >= 1e9
        ? ([1e9, 'B'] as const)
        : abs >= 1e6
          ? ([1e6, 'M'] as const)
          : abs >= 1e3
            ? ([1e3, 'k'] as const)
            : ([1, ''] as const);

  const text = (value / div).toFixed(suffix ? 1 : 2);
  return `${text.includes('.') ? text.replace(/\.?0+$/, '') : text}${suffix}`;
}

/** "1 decision" / "3 decisions". */
export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`;
}

/** NEWS_SENTIMENT -> "news sentiment", for prose contexts. */
export function humanise(token: string): string {
  return token.toLowerCase().replace(/_/g, ' ');
}
