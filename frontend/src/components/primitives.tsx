// ═══════════════════════════════════════════════════════
// FinSight web — shared primitives
// ═══════════════════════════════════════════════════════
//
// The handful of elements every view repeats. Small enough to live in one
// file; each is a thin wrapper over the classes in tokens.css rather than a
// component with opinions of its own.
// ═══════════════════════════════════════════════════════

import type { CSSProperties, ReactNode } from 'react';
import { severityInk, severityMark } from '../lib/format';

/** A wireframe panel with registration marks at its corners. */
export function Blueprint({
  children,
  style,
  id,
}: {
  children: ReactNode;
  style?: CSSProperties;
  id?: string;
}) {
  return (
    <div className="blueprint" style={style} id={id}>
      <i className="corner tl" />
      <i className="corner tr" />
      <i className="corner bl" />
      <i className="corner br" />
      {children}
    </div>
  );
}

/**
 * A severity badge — marker first, colour second.
 *
 * The marker is not decorative. Colour is reinforcement only: strip it and
 * `[!!] HIGH` still reads. See severityMark's note.
 */
export function SeverityMark({ severity, boxed = false }: { severity: string; boxed?: boolean }) {
  const s = severity?.toUpperCase() ?? '?';
  const ink = severityInk(s);
  return (
    <span
      className="mono"
      style={{
        fontSize: 12.5,
        letterSpacing: '0.07em',
        color: ink,
        ...(boxed
          ? { padding: '2px 7px', border: `1px solid ${s === 'HIGH' ? 'var(--alert)' : 'var(--color-divider)'}` }
          : {}),
      }}
    >
      {severityMark(s)} {s}
    </span>
  );
}

/** An uppercase section label. */
export function Kicker({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div className="kicker" style={{ marginBottom: 12, ...style }}>
      {children}
    </div>
  );
}

/**
 * A failed call, shown in place rather than swallowed.
 *
 * Deliberately not styled as an alert-severity element — a backend that is
 * unreachable is a different kind of problem from a HIGH finding, and reusing
 * the alert colour for both would dilute the one that matters.
 */
export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <p
      className="mono"
      style={{
        fontSize: 12.5,
        lineHeight: 1.65,
        margin: '0 0 20px',
        padding: '10px 14px',
        border: '1px solid var(--color-divider)',
        color: 'var(--ink-70)',
        maxWidth: '68ch',
      }}
    >
      {error}
    </p>
  );
}

/** The first-load placeholder. Deliberately quiet — most loads are fast. */
export function Loading({ label = 'Reading…' }: { label?: string }) {
  return (
    <p className="mono" style={{ fontSize: 12.5, letterSpacing: '0.06em', color: 'var(--ink-45)' }}>
      {label}
    </p>
  );
}

/**
 * A designed empty state.
 *
 * "Nothing is waiting on a decision" is what this product shows most days.
 * The brief calls out that its most common state was its least designed one,
 * so an empty list gets a headline and a sentence, never one grey line.
 */
export function Empty({ headline, children }: { headline: string; children?: ReactNode }) {
  return (
    <div style={{ padding: '38px 0 8px', borderTop: '1px solid var(--color-divider)' }}>
      <p
        style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 600,
          fontSize: 22,
          margin: '0 0 8px',
          letterSpacing: '-0.01em',
        }}
      >
        {headline}
      </p>
      {children ? (
        <p style={{ fontSize: 14.5, lineHeight: 1.6, margin: 0, maxWidth: '58ch', color: 'var(--ink-62)' }}>
          {children}
        </p>
      ) : null}
    </div>
  );
}

/** A thin progress rule. `pct` is 0–1. */
export function Meter({ pct, tone = 'var(--color-accent)', height = 3 }: { pct: number; tone?: string; height?: number }) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(pct) ? pct : 0));
  return (
    <div style={{ height, background: 'var(--color-divider)' }}>
      <div style={{ height, width: `${(clamped * 100).toFixed(1)}%`, background: tone, transition: 'width 0.4s ease' }} />
    </div>
  );
}

/** A page heading with its standing explanation. */
export function PageHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <>
      <h1
        style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 600,
          fontSize: 44,
          lineHeight: 1.05,
          letterSpacing: '-0.02em',
          margin: '0 0 14px',
        }}
      >
        {title}
      </h1>
      {children ? (
        <p style={{ fontSize: 16, maxWidth: '54ch', margin: '0 0 32px', color: 'var(--ink-70)', textWrap: 'pretty' }}>
          {children}
        </p>
      ) : null}
    </>
  );
}
