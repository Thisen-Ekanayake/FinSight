// ═══════════════════════════════════════════════════════
// FinSight web — ambient field (three.js)
// ═══════════════════════════════════════════════════════
//
// Purpose : The "the watcher is running" field behind the Desk hero. A slow-
//           breathing plane of points. Every few seconds a run ignites
//           somewhere and a ring of brightness travels outward through the
//           field, then fades. Held items sit as steady alert-coloured
//           beacons that never stop breathing.
//
// ══ THIS IS NOT A CHART ══
//   Nothing here is a measurement, and it must never become one. The design
//   brief is explicit that there is no live market data — the free-tier
//   sources are delayed and capped — so a surface that appeared to plot
//   something would be dishonest. The one thing it does encode is real and
//   discrete: `beacons.length` is how many decisions are actually waiting.
//
// ══ WHY IT MANAGES ITS OWN LIFETIME VIA canvas.__fsAmbient ══
//   React 19's StrictMode mounts effects twice in development, and a
//   WebGL context is expensive enough that building two and throwing one
//   away shows up as a visible stutter. Stamping the instance on the canvas
//   makes a second mount against the same element a no-op that returns the
//   live instance instead of a rival renderer.
// ═══════════════════════════════════════════════════════

import * as THREE from 'three';
import type { Beacon } from './beacons';

export type { Beacon };

export interface VizColors {
  dim: string;
  accent: string;
  alert: string;
}

export interface AmbientHandle {
  setTheme(colors: VizColors): void;
  setBeacons(beacons: Beacon[]): void;
  /** Ignite a ring at the centre — called when a cycle is triggered by hand. */
  ping(alert?: boolean): void;
  dispose(): void;
}

interface CanvasWithAmbient extends HTMLCanvasElement {
  __fsAmbient?: AmbientHandle;
}

const COLS = 84;
const ROWS = 26;
const SPAN_X = 17;
const SPAN_Y = 5.2;
const N = COLS * ROWS;

export function mountAmbient(
  canvas: HTMLCanvasElement,
  colors: VizColors,
  opts: { beacons?: Beacon[] } = {},
): AmbientHandle {
  const host = canvas as CanvasWithAmbient;
  if (host.__fsAmbient) {
    host.__fsAmbient.setTheme(colors);
    host.__fsAmbient.setBeacons(opts.beacons ?? []);
    return host.__fsAmbient;
  }

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 2, 0.1, 60);
  camera.position.set(0, 1.35, 9.4);
  camera.lookAt(0, -0.15, 0);

  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const base = new Float32Array(N * 2);
  let i = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const x = (c / (COLS - 1) - 0.5) * SPAN_X + (Math.random() - 0.5) * 0.09;
      const y = (r / (ROWS - 1) - 0.5) * SPAN_Y + (Math.random() - 0.5) * 0.09;
      base[i * 2] = x;
      base[i * 2 + 1] = y;
      pos[i * 3] = x;
      pos[i * 3 + 1] = y;
      pos[i * 3 + 2] = 0;
      i++;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.052,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: 0.95,
    depthWrite: false,
  });
  const points = new THREE.Points(geo, mat);
  points.rotation.x = -0.42;
  scene.add(points);

  interface PlacedBeacon {
    x: number;
    y: number;
    phase: number;
  }
  let beacons: PlacedBeacon[] = [];

  function placeBeacons(list: Beacon[]): void {
    beacons = list.map((b, k) => ({
      x: b.x * SPAN_X * 0.5,
      y: b.y * SPAN_Y * 0.5,
      phase: k * 1.7,
    }));
  }
  placeBeacons(opts.beacons ?? []);

  function normalize(c: VizColors) {
    return {
      dim: new THREE.Color(c.dim),
      accent: new THREE.Color(c.accent),
      alert: new THREE.Color(c.alert),
    };
  }
  let theme = normalize(colors);

  interface Pulse {
    x: number;
    y: number;
    t0: number;
    alert: boolean;
  }
  const pulses: Pulse[] = [];
  let nextPulse = 1.2;

  const clock = new THREE.Clock();
  let raf = 0;
  let w = 0;
  let h = 0;

  function resize(force = false): void {
    const cw = canvas.clientWidth || 800;
    const ch = canvas.clientHeight || 320;
    if (!force && cw === w && ch === h) return;
    w = cw;
    h = ch;
    renderer.setSize(cw, ch, false);
    camera.aspect = cw / ch;
    camera.fov = ch > 460 ? 44 : 38;
    camera.updateProjectionMatrix();
  }

  const tmp = new THREE.Color();

  function step(): void {
    resize();
    const t = clock.getElapsedTime();

    if (t > nextPulse) {
      pulses.push({
        x: (Math.random() - 0.5) * SPAN_X * 0.82,
        y: (Math.random() - 0.5) * SPAN_Y * 0.7,
        t0: t,
        alert: Math.random() < 0.16,
      });
      nextPulse = t + 2.1 + Math.random() * 3.4;
      if (pulses.length > 5) pulses.shift();
    }

    const p = geo.attributes.position.array as Float32Array;
    const c = geo.attributes.color.array as Float32Array;

    for (let k = 0; k < N; k++) {
      const x = base[k * 2];
      const y = base[k * 2 + 1];

      // breathing surface
      const z =
        Math.sin(x * 0.42 + t * 0.28) * 0.3 +
        Math.cos(y * 0.85 - t * 0.19) * 0.2 +
        Math.sin((x + y) * 0.24 + t * 0.11) * 0.16;
      p[k * 3 + 2] = z;

      let lift = 0.1 + z * 0.16; // gentle self-shading
      let alertMix = 0;

      for (let q = 0; q < pulses.length; q++) {
        const pu = pulses[q];
        const age = t - pu.t0;
        if (age < 0 || age > 4.6) continue;
        const d = Math.hypot(x - pu.x, y - pu.y);
        const ring = d - age * 2.35;
        const band = Math.exp(-(ring * ring) / 0.2) * (1 - age / 4.6);
        lift += band * 1.35;
        if (pu.alert) alertMix = Math.max(alertMix, band);
      }

      for (let q = 0; q < beacons.length; q++) {
        const b = beacons[q];
        const d = Math.hypot(x - b.x, y - b.y);
        const glow = Math.exp(-(d * d) / 0.55) * (0.55 + 0.45 * Math.sin(t * 1.5 + b.phase));
        lift += glow * 0.9;
        alertMix = Math.max(alertMix, glow);
      }

      tmp.copy(theme.dim).lerp(theme.accent, Math.min(1, Math.max(0, lift)));
      if (alertMix > 0) tmp.lerp(theme.alert, Math.min(1, alertMix));
      const s = 0.34 + Math.min(1.25, Math.max(0, lift)) * 0.66;
      c[k * 3] = tmp.r * s;
      c[k * 3 + 1] = tmp.g * s;
      c[k * 3 + 2] = tmp.b * s;
    }

    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    points.rotation.z = Math.sin(t * 0.05) * 0.014;
    renderer.render(scene, camera);
  }

  function frame(): void {
    try {
      step();
    } catch (exc) {
      // A frame that throws would otherwise take the whole loop down and
      // leave a dead canvas over the most important screen in the product.
      console.warn('ambient frame:', exc);
    }
    raf = requestAnimationFrame(frame);
  }

  // A field that breathes is decorative; someone who asked for less motion
  // gets the same composition rendered once and left alone.
  const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  if (still) {
    resize(true);
    step();
  } else {
    raf = requestAnimationFrame(frame);
  }

  let ro: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    ro = new ResizeObserver(() => {
      resize(true);
      if (still) step();
    });
    ro.observe(canvas);
  }

  const api: AmbientHandle = {
    setTheme(next) {
      theme = normalize(next);
      if (still) step();
    },
    setBeacons(next) {
      placeBeacons(next);
      if (still) step();
    },
    ping(alert = false) {
      pulses.push({ x: 0, y: 0, t0: clock.getElapsedTime(), alert });
    },
    dispose() {
      if (host.__fsAmbient === api) delete host.__fsAmbient;
      cancelAnimationFrame(raf);
      ro?.disconnect();
      geo.dispose();
      mat.dispose();
      renderer.dispose();
    },
  };

  host.__fsAmbient = api;
  return api;
}
