// ═══════════════════════════════════════════════════════
// FinSight web — dedup space mount point
// ═══════════════════════════════════════════════════════
//
// Owns the three.js instance's lifetime against React's, exactly as
// AmbientField does — three is imported dynamically so the ~600 kB never
// blocks first paint, and everything below survives the canvas unmounting
// before the module lands.
//
// ══ THE CANVAS IS NEVER THE ONLY COPY ══
//   A WebGL volume cannot be read by a screen reader, cannot be tabbed
//   through, and does not exist for anyone whose browser or GPU refuses the
//   context. So the counts underneath are not a caption — they are the same
//   information in a form that always works, and they render whether or not
//   the canvas ever does. The 3D view is the richer telling of a fact this
//   component states outright regardless.
// ═══════════════════════════════════════════════════════

import { useEffect, useRef, useState } from 'react';
import type { VizColors } from '../viz/ambient';
import type { DedupPoint, DedupSpaceHandle } from '../viz/dedupSpace';

export function DedupSpace({
  colors,
  points,
  thresholds,
  label,
}: {
  colors: VizColors;
  points: DedupPoint[];
  thresholds: { high: number; low: number };
  label: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const handleRef = useRef<DedupSpaceHandle | null>(null);
  const [failed, setFailed] = useState(false);

  // The latest props, readable inside the async mount without making the
  // effect depend on them — a re-render must not tear down a live context.
  const latest = useRef({ colors, points, thresholds });
  latest.current = { colors, points, thresholds };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let live = true;
    void (async () => {
      try {
        const { mountDedupSpace } = await import('../viz/dedupSpace');
        if (!live || !canvas.isConnected) return;
        handleRef.current = mountDedupSpace(canvas, latest.current.colors, {
          points: latest.current.points,
          thresholds: latest.current.thresholds,
        });
        canvas.style.opacity = '1';
      } catch (exc) {
        // No WebGL, no GPU, or a blocked context. The counts below still tell
        // the story, so this degrades rather than taking the page with it.
        console.warn('dedup space unavailable:', exc);
        if (live) setFailed(true);
      }
    })();

    return () => {
      live = false;
      handleRef.current?.dispose();
      handleRef.current = null;
    };
  }, []);

  useEffect(() => {
    handleRef.current?.setTheme(colors);
  }, [colors]);

  // Serialised so a re-render producing an equal-but-new array does not
  // rebuild every instanced mesh in the scene.
  const dataKey = JSON.stringify({ points, thresholds });
  useEffect(() => {
    const parsed = JSON.parse(dataKey) as { points: DedupPoint[]; thresholds: { high: number; low: number } };
    handleRef.current?.setData(parsed.points, parsed.thresholds);
  }, [dataKey]);

  if (failed) return null;

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={label}
      style={{
        display: 'block',
        width: '100%',
        height: 380,
        opacity: 0,
        transition: 'opacity 0.7s ease',
        cursor: 'grab',
        touchAction: 'none',
      }}
    />
  );
}
