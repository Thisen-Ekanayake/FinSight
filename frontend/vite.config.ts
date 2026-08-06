// ═══════════════════════════════════════════════════════
// FinSight web — Vite config
// ═══════════════════════════════════════════════════════
//
// ══ WHY A DEV PROXY RATHER THAN VITE_API_URL IN DEV ══
//   The backend allows every origin (src/api/main.py), so cross-origin calls
//   from :5173 to :8000 would in fact work. The proxy is here for the ONE
//   endpoint where that stops being true: POST /research/query/stream is
//   Server-Sent Events, and a same-origin path keeps EventSource-shaped
//   buffering behaviour identical between `npm run dev` and the nginx build.
//   Debugging a stream that only stalls in production is not worth the two
//   lines saved.
//
//   In a container build there is no dev server, so the SPA reads
//   VITE_API_URL at build time and falls back to same-origin /api — which
//   nginx.conf proxies to the api service. See frontend/Dockerfile.
// ═══════════════════════════════════════════════════════

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API_TARGET = process.env.API_URL ?? 'http://localhost:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // SSE dies behind a proxy that buffers. Vite's http-proxy does not
        // buffer by default, but compression would re-introduce it.
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => proxyReq.setHeader('Accept-Encoding', 'identity'));
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
