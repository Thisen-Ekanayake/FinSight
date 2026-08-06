// ═══════════════════════════════════════════════════════
// FinSight web — beacon placement
// ═══════════════════════════════════════════════════════
//
// Separate from ambient.ts, which imports three. A view that only needs to
// say WHERE the beacons go must not drag a graphics library into the main
// bundle to do it — see the note in AmbientField about why three is loaded
// on demand. This file has no dependencies at all.
// ═══════════════════════════════════════════════════════

export interface Beacon {
  /** −1…1, a fraction of the field's half-width. */
  x: number;
  /** −1…1, a fraction of the field's half-height. */
  y: number;
}

/**
 * Lay out one beacon per waiting decision, up to six.
 *
 * Deterministic rather than random: the same number of pending decisions
 * always produces the same arrangement, so the 45-second poll behind the
 * header badge does not make the field jump every time it re-renders.
 */
export function beaconsFor(count: number): Beacon[] {
  const shown = Math.min(count, 6);
  if (shown <= 0) return [];
  if (shown === 1) return [{ x: 0, y: 0.1 }];
  return Array.from({ length: shown }, (_, k) => ({
    x: -0.52 + (k * 1.04) / (shown - 1),
    y: k % 2 ? -0.26 : 0.24,
  }));
}
