// ═══════════════════════════════════════════════════════
// FinSight web — plot primitives
// ═══════════════════════════════════════════════════════
//
// Purpose : The handful of chart shapes this dashboard repeats — a trend over
//           reporting periods, a distribution, and a labelled bar run.
//
// ══ WHY NO CHARTING LIBRARY ══
//   The bundle already pays ~600 kB for three.js on one route, and
//   AmbientField goes to real trouble to defer even that until after first
//   paint. A second graphics dependency, for what is ultimately polylines and
//   rectangles, would not earn its weight. Everything here is plain SVG
//   geometry over tokens.css variables, so both themes, high-contrast mode and
//   printing come free rather than needing a theme adapter.
//
// ══ COLOUR REPEATS, IT NEVER CARRIES ══
//   lib/format.ts sets this rule for severity markers and it holds for every
//   mark on these axes: a second series is told apart by its dash pattern AND
//   its point marker, with hue as reinforcement only. In greyscale, or to a
//   viewer with a colour-vision deficiency, two lines still read as two lines.
//
// ══ WHAT THESE ARE ALLOWED TO PLOT ══
//   viz/ambient.ts refuses to encode a measurement it does not have. The
//   inverse obligation lives here: every value on these axes is a figure the
//   backend actually returned. Callers label provider and period alongside,
//   because a filed EDGAR figure and a yfinance fallback are not the same kind
//   of fact and a shared axis would imply they were.
// ═══════════════════════════════════════════════════════

import type { ReactNode } from 'react';

// The internal coordinate system. The SVG scales to its container by viewBox,
// so these are aspect-ratio units rather than pixels — text scales with the
// chart instead of needing a resize observer.
const W = 720;

// Right and left have to clear a whole tick label, not just the plot line:
// both outer x labels are centre-anchored on the axis ends, and the y labels
// are right-anchored 8 units outside it. Too little here and "2025 FY" loses
// its last character off the edge of the viewBox.
const PAD = { top: 16, right: 34, bottom: 34, left: 62 };

/** Series are told apart by all three of these at once, never by hue alone. */
const STROKES = ['var(--color-accent-700)', 'var(--alert)', 'var(--color-accent-400)', 'var(--ink-58)'];
const DASHES = ['', '5 3', '2 3', '8 3 2 3'];
const MARKERS = ['circle', 'square', 'triangle', 'diamond'] as const;

export type MarkerShape = (typeof MARKERS)[number];

export interface PlotPoint {
  /** Categorical x position, e.g. "FY2025". */
  label: string;
  value: number;
}

export interface PlotSeries {
  name: string;
  points: PlotPoint[];
}

/**
 * Round a raw range out to readable tick values.
 *
 * A financial axis that reads 46.23 / 47.61 / 48.99 is arithmetically correct
 * and useless — the eye cannot subtract those. This walks up 1/2/5 x 10^n
 * until the interval is round, which is what makes gridlines skimmable.
 */
export function niceTicks(min: number, max: number, count = 4): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];

  // A flat series has no range to divide. Give it one so the line lands
  // mid-plot rather than on the floor, which would read as a value of zero.
  if (min === max) {
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    min -= pad;
    max += pad;
  }

  const raw = (max - min) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;

  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;

  const ticks: number[] = [];
  // Epsilon guards the float accumulation from dropping the last tick.
  for (let t = start; t <= end + step * 1e-9; t += step) {
    ticks.push(Number(t.toFixed(10)));
  }
  return ticks;
}

/** A point marker whose SHAPE identifies the series, not only its colour. */
function Marker({ shape, x, y, fill, r = 3.2 }: { shape: MarkerShape; x: number; y: number; fill: string; r?: number }) {
  if (shape === 'square') return <rect x={x - r} y={y - r} width={r * 2} height={r * 2} fill={fill} />;
  if (shape === 'triangle') {
    return <polygon points={`${x},${y - r * 1.2} ${x + r * 1.1},${y + r} ${x - r * 1.1},${y + r}`} fill={fill} />;
  }
  if (shape === 'diamond') {
    return <polygon points={`${x},${y - r * 1.3} ${x + r * 1.3},${y} ${x},${y + r * 1.3} ${x - r * 1.3},${y}`} fill={fill} />;
  }
  return <circle cx={x} cy={y} r={r} fill={fill} />;
}

const tickTextProps = {
  fill: 'var(--ink-52)',
  fontSize: 11,
  fontFamily: 'var(--mono)',
} as const;

/**
 * The shared frame: gridlines, y ticks, x labels, and the baseline rule.
 *
 * Charts differ in what they draw INSIDE the plot area, not around it, so the
 * axes live here once and each chart supplies the marks.
 */
function Frame({
  height,
  ticks,
  labels,
  scaleY,
  formatTick,
  children,
  title,
}: {
  height: number;
  ticks: number[];
  labels: string[];
  scaleY: (v: number) => number;
  formatTick: (v: number) => string;
  children: ReactNode;
  title: string;
}) {
  const plotW = W - PAD.left - PAD.right;
  const step = labels.length > 1 ? plotW / (labels.length - 1) : 0;

  return (
    // width/height are the INTRINSIC size, in the viewBox's own units, and the
    // CSS below scales it to the container. height="auto" is not a valid SVG
    // attribute value: the element ends up with no layout height, and with
    // overflow visible the chart then paints outside its own box and over
    // whatever follows it. Giving real dimensions and letting CSS do the
    // scaling is what makes the aspect ratio resolve.
    <svg
      viewBox={`0 0 ${W} ${height}`}
      width={W}
      height={height}
      role="img"
      aria-label={title}
      style={{ display: 'block', width: '100%', height: 'auto' }}
    >
      <title>{title}</title>

      {ticks.map((t) => {
        const y = scaleY(t);
        return (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y} y2={y} stroke="var(--color-divider)" strokeWidth={1} />
            <text x={PAD.left - 8} y={y + 3.5} textAnchor="end" {...tickTextProps}>
              {formatTick(t)}
            </text>
          </g>
        );
      })}

      {labels.map((label, i) => (
        <text
          key={`${label}-${i}`}
          x={labels.length > 1 ? PAD.left + i * step : PAD.left + plotW / 2}
          y={height - PAD.bottom + 18}
          textAnchor="middle"
          {...tickTextProps}
        >
          {label}
        </text>
      ))}

      {children}
    </svg>
  );
}

/**
 * A trend over categorical positions — reporting periods, or runs in order.
 *
 * A single point renders as a marker rather than a line, because a one-period
 * "trend" is a reading and drawing a line through it would invent a slope.
 */
export function LineChart({
  series,
  height = 240,
  format = (v) => String(v),
  title,
}: {
  series: PlotSeries[];
  height?: number;
  format?: (v: number) => string;
  title: string;
}) {
  const labels = series[0]?.points.map((p) => p.label) ?? [];
  const values = series.flatMap((s) => s.points.map((p) => p.value)).filter(Number.isFinite);
  if (!labels.length || !values.length) return null;

  const ticks = niceTicks(Math.min(...values), Math.max(...values));
  const lo = ticks[0] ?? 0;
  const hi = ticks[ticks.length - 1] ?? 1;

  const plotW = W - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const scaleY = (v: number) => PAD.top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;
  const scaleX = (i: number) => (labels.length > 1 ? PAD.left + (i * plotW) / (labels.length - 1) : PAD.left + plotW / 2);

  return (
    <Frame height={height} ticks={ticks} labels={labels} scaleY={scaleY} formatTick={format} title={title}>
      {series.map((s, si) => {
        const stroke = STROKES[si % STROKES.length];
        const marker = MARKERS[si % MARKERS.length];
        const drawn = s.points.map((p, i) => ({ x: scaleX(i), y: scaleY(p.value), ok: Number.isFinite(p.value) })).filter((p) => p.ok);

        return (
          <g key={s.name}>
            {drawn.length > 1 ? (
              <polyline
                points={drawn.map((p) => `${p.x},${p.y}`).join(' ')}
                fill="none"
                stroke={stroke}
                strokeWidth={1.6}
                strokeDasharray={DASHES[si % DASHES.length]}
                strokeLinejoin="round"
              />
            ) : null}
            {drawn.map((p, i) => (
              <Marker key={i} shape={marker} x={p.x} y={p.y} fill={stroke} />
            ))}
          </g>
        );
      })}
    </Frame>
  );
}

/** Vertical bars over categorical positions. */
export function BarChart({
  points,
  height = 240,
  format = (v) => String(v),
  title,
  tone = 'var(--color-accent-600)',
}: {
  points: PlotPoint[];
  height?: number;
  format?: (v: number) => string;
  title: string;
  tone?: string;
}) {
  const values = points.map((p) => p.value).filter(Number.isFinite);
  if (!points.length || !values.length) return null;

  // Bars are read as area from a baseline, so the axis must include zero —
  // a truncated bar axis exaggerates differences and is the classic way to
  // mislead with an otherwise honest number.
  const ticks = niceTicks(Math.min(0, ...values), Math.max(0, ...values));
  const lo = ticks[0] ?? 0;
  const hi = ticks[ticks.length - 1] ?? 1;

  const plotW = W - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;
  const scaleY = (v: number) => PAD.top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;

  const slot = plotW / points.length;
  const barW = Math.min(38, slot * 0.62);
  const zero = scaleY(0);

  return (
    <Frame
      height={height}
      ticks={ticks}
      labels={points.map((p) => p.label)}
      scaleY={scaleY}
      formatTick={format}
      title={title}
    >
      {points.map((p, i) => {
        if (!Number.isFinite(p.value)) return null;
        const y = scaleY(p.value);
        const x = PAD.left + i * slot + slot / 2 - barW / 2;
        return (
          <rect
            key={`${p.label}-${i}`}
            x={x}
            y={Math.min(y, zero)}
            width={barW}
            height={Math.max(1, Math.abs(zero - y))}
            fill={tone}
          />
        );
      })}
    </Frame>
  );
}

export interface FunnelRow {
  label: string;
  value: number;
  /** Optional note shown after the value, e.g. "of 12 candidates". */
  note?: string;
  tone?: string;
}

/**
 * Labelled horizontal bars.
 *
 * Used where the categories have names longer than a tick label can carry —
 * a monitoring cycle's candidate/fired/suppressed/merged funnel, or a
 * severity breakdown. Widths are relative to the largest row, and the number
 * is always printed: the bar is the comparison, the figure is the fact.
 */
export function FunnelBars({ rows }: { rows: FunnelRow[] }) {
  const max = Math.max(...rows.map((r) => (Number.isFinite(r.value) ? r.value : 0)), 1);

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {rows.map((row) => (
        <div key={row.label}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'baseline',
              gap: 12,
              marginBottom: 4,
            }}
          >
            <span className="mono" style={{ fontSize: 12, letterSpacing: '0.06em', color: 'var(--ink-70)' }}>
              {row.label}
            </span>
            <span className="mono" style={{ fontSize: 12, color: 'var(--ink-52)' }}>
              {row.value.toLocaleString()}
              {row.note ? ` ${row.note}` : ''}
            </span>
          </div>
          <div style={{ height: 6, background: 'var(--color-divider)' }}>
            <div
              style={{
                height: 6,
                width: `${((Number.isFinite(row.value) ? row.value : 0) / max) * 100}%`,
                background: row.tone ?? 'var(--color-accent-600)',
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * The key for a multi-series chart.
 *
 * Renders the same marker shape and dash the chart drew, so the legend is
 * readable without colour — see this module's header.
 */
export function PlotLegend({ names }: { names: string[] }) {
  if (names.length < 2) return null;

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginTop: 10 }}>
      {names.map((name, i) => (
        <span key={name} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <svg width={26} height={10} aria-hidden="true">
            <line
              x1={0}
              x2={26}
              y1={5}
              y2={5}
              stroke={STROKES[i % STROKES.length]}
              strokeWidth={1.6}
              strokeDasharray={DASHES[i % DASHES.length]}
            />
            <Marker shape={MARKERS[i % MARKERS.length]} x={13} y={5} fill={STROKES[i % STROKES.length]} />
          </svg>
          <span className="mono" style={{ fontSize: 11.5, color: 'var(--ink-62)' }}>
            {name}
          </span>
        </span>
      ))}
    </div>
  );
}

/**
 * The line under a chart naming where its numbers came from.
 *
 * Not decoration. The fundamentals chain falls through EDGAR -> yfinance ->
 * FMP, and a chart that did not say which one served a point would present a
 * fallback estimate with the same authority as a filed figure.
 */
export function PlotSource({ children }: { children: ReactNode }) {
  return (
    <p
      className="mono"
      style={{ fontSize: 11, letterSpacing: '0.05em', color: 'var(--ink-45)', margin: '10px 0 0' }}
    >
      {children}
    </p>
  );
}
