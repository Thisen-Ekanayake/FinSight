// ═══════════════════════════════════════════════════════
// FinSight web — fintech UI kit
// ═══════════════════════════════════════════════════════
//
// The interactive building blocks the finance views share: cards, animated
// progress bars and rings, count-up stats, sliders, ticker chips, toasts and a
// small icon set. Thin wrappers over tokens.css classes, like primitives.tsx.
// ═══════════════════════════════════════════════════════

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
} from 'react';

// ── Icons ───────────────────────────────────────────────

const PATHS: Record<string, string> = {
  home: 'M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z',
  wallet: 'M3 7a2 2 0 0 1 2-2h13v4M3 7v11a2 2 0 0 0 2 2h15V9H5a2 2 0 0 1-2-2zm14 7h.01',
  chart: 'M4 19V5m0 14h16M8 15l3-4 3 2 5-6',
  brain:
    'M9 4a3 3 0 0 0-3 3v.5A3 3 0 0 0 4 10.3a3 3 0 0 0 1 5.2V16a3 3 0 0 0 4 2.8M9 4a3 3 0 0 1 3 3v12m-3-15v0m6 0a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 2.8 3 3 0 0 1-1 5.2V16a3 3 0 0 1-4 2.8M15 4a3 3 0 0 0-3 3',
  inbox: 'M3 13h5l1.5 3h5L16 13h5M5 5h14l2 8v6H3v-6z',
  search: 'M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zm9 3-4.3-4.3',
  flag: 'M5 21V4m0 0h11l-2 4 2 4H5',
  eye: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zm10 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  gear: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7 7 0 0 0-2-1.2L14.5 3h-5l-.4 2.6a7 7 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7 7 0 0 0 2 1.2l.4 2.6h5l.4-2.6a7 7 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2z',
  plus: 'M12 5v14M5 12h14',
  trash: 'M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3',
  play: 'M7 5v14l12-7z',
  check: 'M5 12.5 10 17l9-10',
  x: 'M6 6l12 12M18 6 6 18',
  left: 'M15 6l-6 6 6 6',
  right: 'M9 6l6 6-6 6',
  up: 'M12 19V5m-6 6 6-6 6 6',
  down: 'M12 5v14m6-6-6 6-6-6',
  target: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zm0-4a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0-4a1 1 0 1 0 0-2 1 1 0 0 0 0 2z',
  shield: 'M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z',
  link: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1',
  bolt: 'M13 2 4 14h7l-1 8 9-12h-7z',
  sun: 'M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0-15v2m0 16v2M4.2 4.2l1.4 1.4m12.8 12.8 1.4 1.4M2 12h2m16 0h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4',
  moon: 'M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z',
  logout: 'M15 4h4v16h-4M10 8l-4 4 4 4m-4-4h11',
  sparkle: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z',
  pause: 'M8 5v14m8-14v14',
};

export function Icon({ name, size = 18, style }: { name: keyof typeof PATHS | string; size?: number; style?: CSSProperties }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ flex: 'none', ...style }}
    >
      <path d={PATHS[name] ?? PATHS.sparkle} />
    </svg>
  );
}

// ── Layout ──────────────────────────────────────────────

export function Card({
  title,
  action,
  children,
  style,
  hover = false,
  className = '',
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  style?: CSSProperties;
  hover?: boolean;
  className?: string;
}) {
  return (
    <section className={`card rise ${hover ? 'card-hover' : ''} ${className}`} style={style}>
      {title || action ? (
        <div className="card-title">
          {typeof title === 'string' ? <h3>{title}</h3> : title}
          {action}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function PageTitle({ title, subtitle, action }: { title: string; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    <div className="row between wrap-row" style={{ marginBottom: 22, alignItems: 'flex-end' }}>
      <div>
        <h1 style={{ margin: 0 }}>{title}</h1>
        {subtitle ? <p style={{ margin: '6px 0 0', color: 'var(--ink-58)', maxWidth: '70ch' }}>{subtitle}</p> : null}
      </div>
      {action}
    </div>
  );
}

export function Skeleton({ height = 16, width = '100%', style }: { height?: number; width?: number | string; style?: CSSProperties }) {
  return <div className="skeleton" style={{ height, width, ...style }} />;
}

// ── Progress ────────────────────────────────────────────

/** Animated bar. Grows from zero on mount so the eye catches the amount. */
export function ProgressBar({
  value,
  max = 1,
  tone,
  height = 8,
  striped = false,
  label,
  right,
}: {
  value: number;
  max?: number;
  tone?: string;
  height?: number;
  striped?: boolean;
  label?: ReactNode;
  right?: ReactNode;
}) {
  const target = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(target));
    return () => cancelAnimationFrame(id);
  }, [target]);

  return (
    <div>
      {label || right ? (
        <div className="row between" style={{ fontSize: 13, marginBottom: 6, gap: 8 }}>
          <span style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{label}</span>
          <span className="num" style={{ color: 'var(--ink-58)', flex: 'none' }}>
            {right}
          </span>
        </div>
      ) : null}
      <div
        className={`progress ${striped ? 'striped' : ''}`}
        style={{ height, ['--tone' as string]: tone }}
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(target * 100)}
      >
        <div className="fill" style={{ width: `${(shown * 100).toFixed(2)}%` }} />
      </div>
    </div>
  );
}

/** Circular progress with a centred label. */
export function Ring({
  value,
  size = 96,
  stroke = 10,
  tone = 'var(--color-accent)',
  children,
}: {
  value: number;
  size?: number;
  stroke?: number;
  tone?: string;
  children?: ReactNode;
}) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(clamped));
    return () => cancelAnimationFrame(id);
  }, [clamped]);
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div style={{ position: 'relative', width: size, height: size, flex: 'none' }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--color-field)" strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={tone}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - shown)}
          style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.2, 0.8, 0.2, 1)' }}
        />
      </svg>
      <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', textAlign: 'center' }}>
        {children}
      </div>
    </div>
  );
}

// ── Numbers ─────────────────────────────────────────────

/** Ease a number from its previous value to `value`. */
export function useCountUp(value: number, ms = 700): number {
  const [shown, setShown] = useState(value);
  const from = useRef(0);
  useEffect(() => {
    const start = performance.now();
    const origin = from.current;
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      setShown(origin + (value - origin) * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
      else from.current = value;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, ms]);
  return shown;
}

export function Stat({
  label,
  value,
  format,
  icon,
  tone = 'var(--color-accent)',
  footer,
}: {
  label: string;
  value: number;
  format: (n: number) => string;
  icon?: string;
  tone?: string;
  footer?: ReactNode;
}) {
  const shown = useCountUp(value);
  return (
    <div className="card card-hover rise" style={{ padding: 18 }}>
      <div className="row between" style={{ marginBottom: 10 }}>
        <span style={{ fontSize: 13, color: 'var(--ink-58)', fontWeight: 500 }}>{label}</span>
        {icon ? (
          <span
            style={{
              display: 'grid',
              placeItems: 'center',
              width: 32,
              height: 32,
              borderRadius: 10,
              color: tone,
              background: `color-mix(in srgb, ${tone} 13%, transparent)`,
            }}
          >
            <Icon name={icon} size={17} />
          </span>
        ) : null}
      </div>
      <div className="num" style={{ fontFamily: 'var(--font-heading)', fontSize: 26, fontWeight: 600, letterSpacing: '-0.02em' }}>
        {format(shown)}
      </div>
      {footer ? <div style={{ marginTop: 6, fontSize: 12.5, color: 'var(--ink-58)' }}>{footer}</div> : null}
    </div>
  );
}

export function Delta({ value, format }: { value: number; format: (n: number) => string }) {
  const up = value > 0;
  const flat = value === 0;
  return (
    <span className={`chip ${flat ? '' : up ? 'chip-pos' : 'chip-neg'}`}>
      {flat ? null : <Icon name={up ? 'up' : 'down'} size={12} />}
      {format(value)}
    </span>
  );
}

// ── Inputs ──────────────────────────────────────────────

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (n) => String(n),
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
  format?: (n: number) => string;
  hint?: ReactNode;
}) {
  const fill = ((value - min) / (max - min)) * 100;
  return (
    <div className="field">
      <div className="row between" style={{ marginBottom: 8 }}>
        <label style={{ margin: 0 }}>{label}</label>
        <span className="chip chip-accent num">{format(value)}</span>
      </div>
      <input
        className="slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ['--fill' as string]: `${fill}%` }}
      />
      {hint ? <div style={{ fontSize: 12, color: 'var(--ink-52)', marginTop: 6 }}>{hint}</div> : null}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className="switch" style={{ opacity: disabled ? 0.5 : 1 }}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="track" />
      {label}
    </label>
  );
}

/** Ticker chips: type, Enter/comma to add, Backspace to remove the last. */
export function TickerInput({
  value,
  onChange,
  max = 8,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  max?: number;
}) {
  const [draft, setDraft] = useState('');
  const add = (raw: string) => {
    const symbols = raw
      .split(/[\s,]+/)
      .map((s) => s.trim().toUpperCase())
      .filter((s) => /^[A-Z.-]{1,10}$/.test(s));
    const next = Array.from(new Set([...value, ...symbols])).slice(0, max);
    onChange(next);
    setDraft('');
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',' || e.key === ' ') {
      e.preventDefault();
      if (draft.trim()) add(draft);
    } else if (e.key === 'Backspace' && !draft && value.length) {
      onChange(value.slice(0, -1));
    }
  };
  return (
    <div
      className="input"
      style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center', height: 'auto', padding: 6 }}
    >
      {value.map((t, i) => (
        <span key={t} className="chip chip-accent pop" style={{ ['--i' as string]: i }}>
          {t}
          <button type="button" aria-label={`Remove ${t}`} onClick={() => onChange(value.filter((x) => x !== t))}>
            ×
          </button>
        </span>
      ))}
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={onKey}
        onBlur={() => draft.trim() && add(draft)}
        placeholder={value.length >= max ? `Max ${max}` : 'Add ticker…'}
        disabled={value.length >= max}
        aria-label="Add ticker"
        style={{ flex: 1, minWidth: 90, border: 0, background: 'transparent', font: 'inherit', color: 'inherit', outline: 'none', padding: 4 }}
      />
    </div>
  );
}

// ── Toasts ──────────────────────────────────────────────

type ToastTone = 'info' | 'success' | 'error';
interface ToastItem {
  id: number;
  text: string;
  tone: ToastTone;
}

const ToastContext = createContext<(text: string, tone?: ToastTone) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const push = useCallback((text: string, tone: ToastTone = 'info') => {
    const id = Date.now() + Math.random();
    setItems((xs) => [...xs, { id, text, tone }]);
    window.setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), 4200);
  }, []);
  const toneVar = { info: 'var(--color-accent)', success: 'var(--pos)', error: 'var(--neg)' };
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className="toast" style={{ ['--tone' as string]: toneVar[t.tone] }}>
            <Icon name={t.tone === 'error' ? 'x' : t.tone === 'success' ? 'check' : 'sparkle'} size={16} style={{ color: toneVar[t.tone], marginTop: 2 }} />
            <span>{t.text}</span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}

/** Run an async action with a busy flag and a toast on failure. */
export function useAction() {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const run = useCallback(
    async <T,>(fn: () => Promise<T>, success?: string): Promise<T | undefined> => {
      setBusy(true);
      try {
        const out = await fn();
        if (success) toast(success, 'success');
        return out;
      } catch (exc) {
        toast(exc instanceof Error ? exc.message.replace(/^[A-Z]+ \S+ -> \d+: /, '') : String(exc), 'error');
        return undefined;
      } finally {
        setBusy(false);
      }
    },
    [toast],
  );
  return { busy, run };
}

export function Disclaimer({ children }: { children?: ReactNode }) {
  return (
    <div
      className="row"
      style={{
        gap: 10,
        alignItems: 'flex-start',
        fontSize: 12.5,
        color: 'var(--ink-58)',
        padding: '10px 14px',
        borderRadius: 12,
        background: 'var(--warn-soft)',
      }}
    >
      <Icon name="shield" size={16} style={{ color: 'var(--warn)', marginTop: 1 }} />
      <span>
        {children ??
          'Educational tool, not financial advice. Agents learn from past prices only, and every order goes to an Alpaca paper account (simulated money) after your approval.'}
      </span>
    </div>
  );
}
