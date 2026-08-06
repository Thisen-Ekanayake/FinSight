// ═══════════════════════════════════════════════════════
// FinSight web — theme
// ═══════════════════════════════════════════════════════
//
// Three settings, two outcomes. "auto" follows the OS; "light" and "dark"
// override it. The resolved outcome is written to <html data-theme> because
// tokens.css hangs every dark rule off that attribute — a media query alone
// could not be overridden by an explicit choice. `colorScheme` is set too so
// native form controls and scrollbars follow.
//
// Dark mode is expected, not optional (design brief), so the setting persists
// across reloads.
// ═══════════════════════════════════════════════════════

import { useCallback, useEffect, useState } from 'react';
import type { VizColors } from '../viz/ambient';

export type ThemeSetting = 'light' | 'auto' | 'dark';

const STORAGE_KEY = 'finsight.theme';

function stored(): ThemeSetting {
  const value = localStorage.getItem(STORAGE_KEY);
  return value === 'light' || value === 'dark' || value === 'auto' ? value : 'auto';
}

function prefersDark(): boolean {
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function useTheme(): {
  theme: ThemeSetting;
  setTheme: (theme: ThemeSetting) => void;
  dark: boolean;
  vizColors: VizColors;
} {
  const [theme, setThemeState] = useState<ThemeSetting>(stored);
  const [dark, setDark] = useState<boolean>(() => (stored() === 'auto' ? prefersDark() : stored() === 'dark'));

  useEffect(() => {
    const resolved = theme === 'dark' || (theme === 'auto' && prefersDark());
    document.documentElement.setAttribute('data-theme', resolved ? 'dark' : 'light');
    document.documentElement.style.colorScheme = resolved ? 'dark' : 'light';
    setDark(resolved);
  }, [theme]);

  // Only "auto" listens to the OS. Re-subscribing on every theme change is
  // cheap and keeps the listener from firing while an explicit choice is set.
  useEffect(() => {
    if (theme !== 'auto') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => {
      document.documentElement.setAttribute('data-theme', mq.matches ? 'dark' : 'light');
      document.documentElement.style.colorScheme = mq.matches ? 'dark' : 'light';
      setDark(mq.matches);
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [theme]);

  const setTheme = useCallback((next: ThemeSetting) => {
    localStorage.setItem(STORAGE_KEY, next);
    setThemeState(next);
  }, []);

  // WebGL cannot read a CSS custom property, so the field's palette is the
  // one place the token values have to be repeated in JS. These are the same
  // hexes tokens.css uses for --color-accent and --alert in each theme.
  const vizColors: VizColors = dark
    ? { dim: '#2b333a', accent: '#86aad0', alert: '#d9a24e' }
    : { dim: '#d5d9dd', accent: '#5980a6', alert: '#b07a24' };

  return { theme, setTheme, dark, vizColors };
}
