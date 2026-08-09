// ═══════════════════════════════════════════════════════
// FinSight web — dedup decision space (three.js)
// ═══════════════════════════════════════════════════════
//
// Purpose : The dedup engine's decision boundary, as a volume you can turn.
//           Every candidate it judged sits at (when, how similar, which
//           ticker), and the two thresholds are the planes cutting through
//           them. Where a point sits relative to a plane IS the decision.
//
// ══ WHY THIS ONE IS ACTUALLY 3D ══
//   viz/ambient.ts refuses to encode measurements, and most of this
//   dashboard's data is flatly 2D — a margin over three fiscal years wants a
//   line, and rendering it as a surface would cost readability to buy nothing.
//   This dataset is the exception. Similarity, time and ticker are three
//   independent axes, and the question people actually ask of a dedup log —
//   "how close to the line was that call?" — is a question about distance to
//   a plane. Flattened to 2D the planes become two horizontal rules and the
//   per-ticker structure collapses into overplotting.
//
// ══ COLOUR REPEATS, SHAPE CARRIES ══
//   Same rule as viz/plot.tsx and lib/format.ts. Each outcome gets its own
//   GEOMETRY — octahedron reported, cube folded, tetrahedron held for a second
//   look — so the volume survives greyscale and colour-vision deficiency.
//   Hue only ever restates the shape.
//
// ══ WHY IT MANAGES ITS OWN LIFETIME VIA canvas.__fsDedup ══
//   Identical reasoning to ambient.ts: React 19 StrictMode mounts effects
//   twice in development and a WebGL context is far too expensive to build
//   and discard. Stamping the instance on the canvas makes the second mount
//   return the live one instead of a rival renderer.
// ═══════════════════════════════════════════════════════

import * as THREE from 'three';
import type { VizColors } from './ambient';

/** One judged candidate, already normalised to 0–1 on every axis. */
export interface DedupPoint {
  /** Position in the log's time range, oldest 0 to newest 1. */
  t: number;
  /** Similarity to the alert it was compared against. */
  score: number;
  /** Which ticker lane, spread evenly across the depth axis. */
  lane: number;
  kind: 'reported' | 'folded' | 'second-look';
}

export interface DedupSpaceHandle {
  setTheme(colors: VizColors): void;
  setData(points: DedupPoint[], thresholds: { high: number; low: number }): void;
  dispose(): void;
}

interface CanvasWithSpace extends HTMLCanvasElement {
  __fsDedup?: DedupSpaceHandle;
}

const SPAN_X = 6.4;
const SPAN_Y = 3.4;
const SPAN_Z = 4.0;

const KINDS = ['reported', 'folded', 'second-look'] as const;

/** Geometry per outcome. The shape is the encoding; colour only repeats it. */
function geometryFor(kind: DedupPoint['kind']): THREE.BufferGeometry {
  if (kind === 'reported') return new THREE.OctahedronGeometry(0.085);
  if (kind === 'second-look') return new THREE.TetrahedronGeometry(0.105);
  return new THREE.BoxGeometry(0.115, 0.115, 0.115);
}

export function mountDedupSpace(
  canvas: HTMLCanvasElement,
  colors: VizColors,
  opts: { points?: DedupPoint[]; thresholds?: { high: number; low: number } } = {},
): DedupSpaceHandle {
  const host = canvas as CanvasWithSpace;
  if (host.__fsDedup) {
    host.__fsDedup.setTheme(colors);
    if (opts.points) host.__fsDedup.setData(opts.points, opts.thresholds ?? { high: 0.89, low: 0.74 });
    return host.__fsDedup;
  }

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 2, 0.1, 100);

  // Lit rather than flat: the shapes only read as distinct solids if their
  // faces catch light differently.
  scene.add(new THREE.AmbientLight(0xffffff, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(4, 7, 5);
  scene.add(key);

  let theme = { ...colors };

  // ── The cage ──
  // A wireframe box around the data volume, echoing the registration-mark
  // panels in tokens.css. Without it the points float in undefined space and
  // rotation gives no sense of where the volume's edges are.
  const cageMat = new THREE.LineBasicMaterial({ color: new THREE.Color(theme.dim), transparent: true, opacity: 0.85 });
  const cage = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(SPAN_X, SPAN_Y, SPAN_Z)),
    cageMat,
  );
  cage.position.y = SPAN_Y / 2;
  scene.add(cage);

  const grid = new THREE.GridHelper(SPAN_X, 8, new THREE.Color(theme.dim), new THREE.Color(theme.dim));
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.35;
  grid.scale.z = SPAN_Z / SPAN_X;
  scene.add(grid);

  // ── The threshold planes ──
  // The payoff of drawing this in 3D at all: the decision boundary as a
  // surface the points sit above or below.
  const planeGeo = new THREE.PlaneGeometry(SPAN_X, SPAN_Z);
  const highMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(theme.accent),
    transparent: true,
    opacity: 0.14,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const lowMat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(theme.alert),
    transparent: true,
    opacity: 0.12,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const highPlane = new THREE.Mesh(planeGeo, highMat);
  const lowPlane = new THREE.Mesh(planeGeo, lowMat);
  highPlane.rotation.x = -Math.PI / 2;
  lowPlane.rotation.x = -Math.PI / 2;
  scene.add(highPlane, lowPlane);

  // ── The points ──
  const clouds = new Map<DedupPoint['kind'], THREE.InstancedMesh>();
  const dummy = new THREE.Object3D();

  function colourFor(kind: DedupPoint['kind']): THREE.Color {
    if (kind === 'second-look') return new THREE.Color(theme.alert);
    if (kind === 'reported') return new THREE.Color(theme.accent);
    return new THREE.Color(theme.dim).lerp(new THREE.Color(theme.accent), 0.45);
  }

  function clearClouds(): void {
    for (const mesh of clouds.values()) {
      scene.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    clouds.clear();
  }

  function setData(points: DedupPoint[], thresholds: { high: number; low: number }): void {
    highPlane.position.y = thresholds.high * SPAN_Y;
    lowPlane.position.y = thresholds.low * SPAN_Y;

    clearClouds();

    for (const kind of KINDS) {
      const subset = points.filter((p) => p.kind === kind);
      if (subset.length === 0) continue;

      const mesh = new THREE.InstancedMesh(
        geometryFor(kind),
        new THREE.MeshLambertMaterial({ color: colourFor(kind) }),
        subset.length,
      );

      subset.forEach((point, i) => {
        dummy.position.set(
          (point.t - 0.5) * SPAN_X,
          THREE.MathUtils.clamp(point.score, 0, 1) * SPAN_Y,
          (point.lane - 0.5) * SPAN_Z,
        );
        dummy.rotation.set(0, i * 0.7, 0);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;

      clouds.set(kind, mesh);
      scene.add(mesh);
    }
  }

  setData(opts.points ?? [], opts.thresholds ?? { high: 0.89, low: 0.74 });

  // ── Orbit ──
  // Hand-rolled rather than pulling three's OrbitControls addon: this needs
  // two angles and a drag, and the addon brings damping, panning and zoom
  // this view has no use for.
  let azimuth = 0.72;
  let elevation = 0.42;
  const RADIUS = 9.6;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let idle = true;

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function onPointerDown(event: PointerEvent): void {
    dragging = true;
    idle = false;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: PointerEvent): void {
    if (!dragging) return;
    azimuth -= (event.clientX - lastX) * 0.006;
    elevation = THREE.MathUtils.clamp(elevation + (event.clientY - lastY) * 0.005, -0.25, 1.25);
    lastX = event.clientX;
    lastY = event.clientY;
  }

  function onPointerUp(event: PointerEvent): void {
    dragging = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  }

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);

  let w = 0;
  let h = 0;

  function resize(): void {
    const cw = canvas.clientWidth || 720;
    const ch = canvas.clientHeight || 380;
    if (cw === w && ch === h) return;
    w = cw;
    h = ch;
    renderer.setSize(cw, ch, false);
    camera.aspect = cw / ch;
    camera.updateProjectionMatrix();
  }

  const clock = new THREE.Clock();
  let raf = 0;

  function step(): void {
    resize();

    // Drifts until touched, then holds wherever it was left — a view that
    // kept spinning under someone reading a cluster would be hostile.
    if (idle && !reduceMotion) azimuth += clock.getDelta() * 0.1;
    else clock.getDelta();

    camera.position.set(
      Math.sin(azimuth) * Math.cos(elevation) * RADIUS,
      SPAN_Y / 2 + Math.sin(elevation) * RADIUS,
      Math.cos(azimuth) * Math.cos(elevation) * RADIUS,
    );
    camera.lookAt(0, SPAN_Y / 2, 0);

    renderer.render(scene, camera);
    raf = requestAnimationFrame(step);
  }

  resize();
  raf = requestAnimationFrame(step);

  const handle: DedupSpaceHandle = {
    setTheme(next: VizColors) {
      theme = { ...next };
      cageMat.color = new THREE.Color(theme.dim);
      (grid.material as THREE.LineBasicMaterial).color = new THREE.Color(theme.dim);
      highMat.color = new THREE.Color(theme.accent);
      lowMat.color = new THREE.Color(theme.alert);
      for (const [kind, mesh] of clouds) {
        (mesh.material as THREE.MeshLambertMaterial).color = colourFor(kind);
      }
    },

    setData,

    dispose() {
      cancelAnimationFrame(raf);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);

      clearClouds();
      cage.geometry.dispose();
      cageMat.dispose();
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      planeGeo.dispose();
      highMat.dispose();
      lowMat.dispose();
      renderer.dispose();

      delete host.__fsDedup;
    },
  };

  host.__fsDedup = handle;
  return handle;
}
