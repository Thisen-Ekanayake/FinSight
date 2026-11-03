// ═══════════════════════════════════════════════════════
// FinSight web — interactive finance charts
// ═══════════════════════════════════════════════════════
//
// TrendChart  multi-series line/area chart with a hover crosshair + tooltip
// Donut       allocation ring with hover-linked legend
//
// Plain SVG, no library (see plot.tsx for why). Width is measured with a
// ResizeObserver so text stays crisp instead of being stretched by viewBox.
// Series differ by dash pattern as well as hue, so they survive greyscale.
// ═══════════════════════════════════════════════════════

import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

export interface TrendSeries {
  name: string;
  values: number[];
  color: string;
  dash?: string;
  area?: boolean;
}

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(600);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export function TrendChart({
  series,
  labels,
  height = 240,
  format = (n) => n.toFixed(2),
  baseline,
}: {
  series: TrendSeries[];
  labels: string[];
  height?: number;
  format?: (n: number) => string;
  /** Draw a dashed reference line at this y value (e.g. 1.0 for "break-even"). */
  baseline?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const pad = { l: 52, r: 12, t: 12, b: 26 };
  const n = Math.max(...series.map((s) => s.values.length), 0);

  const { min, max } = useMemo(() => {
    const all = series.flatMap((s) => s.values).filter(Number.isFinite);
    if (baseline !== undefined) all.push(baseline);
    if (!all.length) return { min: 0, max: 1 };
    let lo = Math.min(...all);
    const allPositive = lo >= 0;
    let hi = Math.max(...all);
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    const padY = (hi - lo) * 0.08;
    // Money that cannot go negative should not get a negative axis.
    return { min: allPositive ? Math.max(0, lo - padY) : lo - padY, max: hi + padY };
  }, [series, baseline]);

  if (n < 2) {
    return (
      <div ref={ref} style={{ height, display: 'grid', placeItems: 'center', color: 'var(--ink-45)', fontSize: 13 }}>
        Not enough data to draw yet.
      </div>
    );
  }

  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;
  const x = (i: number) => pad.l + (i / (n - 1)) * iw;
  const y = (v: number) => pad.t + (1 - (v - min) / (max - min)) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => min + f * (max - min));
  const xTicks = [0, Math.floor((n - 1) / 2), n - 1];

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const i = Math.round(((px - pad.l) / iw) * (n - 1));
    setHover(Math.max(0, Math.min(n - 1, i)));
  };

  const tipLeft = hover !== null ? Math.min(Math.max(x(hover) + 12, 8), width - 190) : 0;

  return (
    <div ref={ref} style={{ position: 'relative', width: '100%' }}>
      <svg
        width={width}
        height={height}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        style={{ display: 'block', touchAction: 'none', cursor: 'crosshair' }}
        role="img"
        aria-label={series.map((s) => s.name).join(', ')}
      >
        <defs>
          {series.map((s, si) => (
            <linearGradient key={si} id={`fs-area-${si}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity={0.28} />
              <stop offset="100%" stopColor={s.color} stopOpacity={0} />
            </linearGradient>
          ))}
        </defs>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="var(--color-divider)" />
            <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" fontSize={11} fill="var(--ink-45)">
              {format(t)}
            </text>
          </g>
        ))}
        {xTicks.map((i) => (
          <text
            key={i}
            x={x(i)}
            y={height - 6}
            textAnchor={i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle'}
            fontSize={11}
            fill="var(--ink-45)"
          >
            {labels[i] ?? ''}
          </text>
        ))}
        {baseline !== undefined ? (
          <line
            x1={pad.l}
            x2={width - pad.r}
            y1={y(baseline)}
            y2={y(baseline)}
            stroke="var(--ink-42)"
            strokeDasharray="3 4"
          />
        ) : null}
        {series.map((s, si) => {
          const pts = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
          const line = `M${pts.join('L')}`;
          return (
            <g key={s.name}>
              {s.area ? (
                <path
                  d={`${line}L${x(s.values.length - 1)},${pad.t + ih}L${x(0)},${pad.t + ih}Z`}
                  fill={`url(#fs-area-${si})`}
                />
              ) : null}
              <path
                d={line}
                fill="none"
                stroke={s.color}
                strokeWidth={si === 0 ? 2.4 : 1.8}
                strokeDasharray={s.dash}
                strokeLinejoin="round"
                strokeLinecap="round"
                className="fs-draw"
              />
            </g>
          );
        })}
        {hover !== null ? (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={pad.t + ih} stroke="var(--ink-42)" />
            {series.map((s) =>
              s.values[hover] !== undefined ? (
                <circle
                  key={s.name}
                  cx={x(hover)}
                  cy={y(s.values[hover])}
                  r={4.5}
                  fill="var(--color-surface)"
                  stroke={s.color}
                  strokeWidth={2.4}
                />
              ) : null,
            )}
          </g>
        ) : null}
      </svg>
      {hover !== null ? (
        <div
          className="card"
          style={{
            position: 'absolute',
            top: 8,
            left: tipLeft,
            padding: '8px 12px',
            pointerEvents: 'none',
            fontSize: 12.5,
            minWidth: 170,
            boxShadow: 'var(--shadow-lg)',
          }}
        >
          <div style={{ color: 'var(--ink-52)', marginBottom: 4 }}>{labels[hover]}</div>
          {series.map((s) =>
            s.values[hover] !== undefined ? (
              <div key={s.name} className="row between" style={{ gap: 14 }}>
                <span className="row" style={{ gap: 6 }}>
                  <svg width={16} height={4} aria-hidden="true">
                    <line x1={0} x2={16} y1={2} y2={2} stroke={s.color} strokeWidth={2.5} strokeDasharray={s.dash} />
                  </svg>
                  {s.name}
                </span>
                <b className="num">{format(s.values[hover])}</b>
              </div>
            ) : null,
          )}
        </div>
      ) : null}
    </div>
  );
}

export function Legend({ series }: { series: Pick<TrendSeries, 'name' | 'color' | 'dash'>[] }) {
  return (
    <div className="row wrap-row" style={{ gap: 16, fontSize: 12.5, color: 'var(--ink-62)', marginTop: 10 }}>
      {series.map((s) => (
        <span key={s.name} className="row" style={{ gap: 6 }}>
          <svg width={20} height={4} aria-hidden="true">
            <line x1={0} x2={20} y1={2} y2={2} stroke={s.color} strokeWidth={3} strokeDasharray={s.dash} />
          </svg>
          {s.name}
        </span>
      ))}
    </div>
  );
}

export interface Slice {
  label: string;
  value: number;
  color: string;
}

export function Donut({
  data,
  size = 180,
  thickness = 26,
  center,
  format = (n) => `${(n * 100).toFixed(1)}%`,
}: {
  data: Slice[];
  size?: number;
  thickness?: number;
  center?: ReactNode;
  format?: (n: number) => string;
}) {
  const [active, setActive] = useState<number | null>(null);
  const total = data.reduce((a, d) => a + Math.max(0, d.value), 0) || 1;
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  let offset = 0;

  return (
    <div className="row wrap-row" style={{ gap: 22, alignItems: 'center' }}>
      <div style={{ position: 'relative', width: size, height: size, flex: 'none' }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }} role="img" aria-label="Allocation">
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-field)" strokeWidth={thickness} />
          {data.map((d, i) => {
            const frac = Math.max(0, d.value) / total;
            const len = frac * c;
            const el = (
              <circle
                key={d.label}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={d.color}
                strokeWidth={active === i ? thickness + 6 : thickness}
                strokeDasharray={`${Math.max(0, len - 2)} ${c}`}
                strokeDashoffset={-offset}
                style={{
                  transition: 'stroke-width 0.15s ease, opacity 0.15s ease',
                  opacity: active === null || active === i ? 1 : 0.35,
                  cursor: 'pointer',
                }}
                onPointerEnter={() => setActive(i)}
                onPointerLeave={() => setActive(null)}
              />
            );
            offset += len;
            return el;
          })}
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', textAlign: 'center' }}>
          {active !== null ? (
            <div>
              <div style={{ fontSize: 12, color: 'var(--ink-58)' }}>{data[active].label}</div>
              <div className="num" style={{ fontFamily: 'var(--font-heading)', fontSize: 22, fontWeight: 600 }}>
                {format(data[active].value / total)}
              </div>
            </div>
          ) : (
            center
          )}
        </div>
      </div>
      <div style={{ display: 'grid', gap: 6, minWidth: 140, flex: 1 }}>
        {data.map((d, i) => (
          <button
            key={d.label}
            type="button"
            onPointerEnter={() => setActive(i)}
            onPointerLeave={() => setActive(null)}
            onFocus={() => setActive(i)}
            onBlur={() => setActive(null)}
            className="row between"
            style={{
              border: 0,
              background: active === i ? 'var(--color-field)' : 'transparent',
              borderRadius: 8,
              padding: '5px 8px',
              font: 'inherit',
              fontSize: 13,
              color: 'inherit',
              cursor: 'default',
            }}
          >
            <span className="row" style={{ gap: 8 }}>
              <span style={{ width: 10, height: 10, borderRadius: 3, background: d.color }} />
              {d.label}
            </span>
            <b className="num">{format(d.value / total)}</b>
          </button>
        ))}
      </div>
    </div>
  );
}

export const SERIES = [
  'var(--series-1)',
  'var(--series-2)',
  'var(--series-3)',
  'var(--series-4)',
  'var(--series-5)',
  'var(--series-6)',
  'var(--series-7)',
  'var(--series-8)',
];
