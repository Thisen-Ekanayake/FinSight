// ═══════════════════════════════════════════════════════
// FinSight web — ambient field mount point
// ═══════════════════════════════════════════════════════
//
// Owns the three.js instance's lifetime against React's. The renderer is
// built once per canvas and then only ever updated — theme and beacon changes
// go through the handle rather than remounting, because rebuilding a WebGL
// context on a theme toggle is both visible and wasteful.
//
// ══ WHY three IS IMPORTED DYNAMICALLY ══
//   three is ~600 kB of the bundle and exists for one decorative canvas on
//   one view. Loading it up front would delay the answer to "does anything
//   need me?" — the only question this screen has three seconds to answer —
//   behind a graphics library. The import starts after first paint and the
//   field fades in when it lands.
//
//   Everything below therefore has to survive the canvas unmounting before
//   the module arrives, which is the normal case for someone who lands on the
//   Desk and immediately navigates away.
// ═══════════════════════════════════════════════════════

import { useEffect, useRef } from 'react';
import type { AmbientHandle, VizColors } from '../viz/ambient';
import type { Beacon } from '../viz/beacons';

export function AmbientField({ colors, beacons }: { colors: VizColors; beacons: Beacon[] }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const handleRef = useRef<AmbientHandle | null>(null);

  // The latest props, readable from inside the async mount without making it
  // depend on them — a re-render must not tear down a live WebGL context.
  const latest = useRef({ colors, beacons });
  latest.current = { colors, beacons };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let live = true;
    void (async () => {
      try {
        const { mountAmbient } = await import('../viz/ambient');
        if (!live || !canvas.isConnected) return;
        handleRef.current = mountAmbient(canvas, latest.current.colors, { beacons: latest.current.beacons });
        canvas.style.opacity = '1';
      } catch (exc) {
        // A dashboard that fails to load because a decorative canvas could not
        // start is a worse product than one with no canvas.
        console.warn('ambient field unavailable:', exc);
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

  // Serialised so a re-render that produced an equal-but-new array does not
  // re-place the beacons and restart their phase.
  const beaconKey = JSON.stringify(beacons);
  useEffect(() => {
    handleRef.current?.setBeacons(JSON.parse(beaconKey) as Beacon[]);
  }, [beaconKey]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{
        position: 'absolute',
        inset: 0,
        display: 'block',
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        opacity: 0,
        transition: 'opacity 0.7s ease',
      }}
    />
  );
}
