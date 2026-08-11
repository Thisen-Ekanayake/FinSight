# FinSight — web dashboard

React 19 + TypeScript + Vite, with one three.js canvas. Talks to the FastAPI
backend over plain HTTP and holds no state of its own: every screen is a read
of the API plus, at most, a form.

```bash
npm install
npm run dev        # localhost:5173, proxies /api -> localhost:8000
```

Or from the repo root: `./shell_scripts/run_web.sh` (starts the dev server and warns if the
API is not up), `make web-build` (type-check + bundle), `make web-check`
(types only).

## What is where

```
src/
  api/         types.ts mirrors src/api/schemas.py; client.ts mirrors src/ui/client.py
  views/       one file per destination — Desk, Ask, Findings, Watchlist, System
  components/  Header, AmbientField, and the primitives every view repeats
  hooks/       useResource (fetch + poll), useTheme, useRoute
  lib/         format.ts (severity markers, elapsed time), answer.tsx (prose + citations)
  viz/         ambient.ts is the three.js field; beacons.ts is its dependency-free geometry
  styles/      tokens.css — the design system, and the source of truth for the look
```

## Five decisions worth knowing

**The Desk is the product.** A HIGH finding stops its run and nothing is
dispatched until a person decides. That interaction used to be page four of a
six-item sidebar; it is now the first thing on the first screen, decided in
place. Reject is preselected for every pending alert and submitting an
untouched form sends nothing — see the note at the top of `views/Desk.tsx`
before changing anything about that.

**Severity is never colour alone.** `[!!] HIGH`, `[~] MED`, `[.] LOW` are
rendered as text; colour only ever repeats what the marker already says. The
failure mode this product cannot afford is someone missing a HIGH, and colour
does not survive greyscale or colour-vision deficiency.

**Progress is real, not a spinner.** A research question takes 30–60 seconds,
so Ask uses `POST /research/query/stream` and shows the specialists arriving
one at a time. The SSE framing is parsed by hand in `api/client.ts` because
`EventSource` is GET-only and the question travels in a POST body. The
progress bar is capped below 100% until the `final` frame lands.

**three.js is loaded on demand.** It is ~470 kB for one decorative canvas on
one view, and the question that view exists to answer — "does anything need
me?" — must not wait behind a graphics library. `AmbientField` imports it
after first paint and fades the canvas in. Nothing in the field is a
measurement; there is no live market data to plot.

**Numbers the UI explains come from the API.** The sentence describing where
the dedup fold line sits reads `dedup_tau_high` / `dedup_tau_low` from
`/admin/config`. They are re-tuned whenever the embedding model changes, and a
hardcoded `0.89` here would quietly start lying.

## Build and deploy

`Dockerfile` is two stages — node builds, nginx serves. `nginx.conf` proxies
`/api` to the `api` service so the browser sees a single origin, with
buffering off and a long read timeout so the SSE stream is not held and
released in one burst. `VITE_API_URL` is deliberately unset in the image: the
bundle falls back to the same-origin `/api` prefix, which keeps one image
usable on any host.

Verify a running backend serves what these types expect:

```bash
python ../scripts/api_smoke.py                  # free
python ../scripts/api_smoke.py --query --cycle  # also spends
```
