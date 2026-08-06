# ═══════════════════════════════════════════════════════
# FinSight web — image
# ═══════════════════════════════════════════════════════
#
# Two stages: node builds the bundle, nginx serves it. The runtime image
# carries no node, no sources, and no build tooling — only static files and a
# web server.
#
# ══ VITE_API_URL IS DELIBERATELY UNSET ══
#   Vite inlines env vars at BUILD time, so a baked-in API URL would make the
#   image specific to one deployment's hostname. Left unset, the bundle falls
#   back to the same-origin /api prefix and nginx.conf decides where that
#   goes — one image, any host.
# ═══════════════════════════════════════════════════════

FROM node:22-alpine AS build

WORKDIR /build

# Dependencies first: this layer is only invalidated by a lockfile change, so
# an ordinary source edit rebuilds in seconds rather than re-resolving npm.
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig*.json vite.config.ts index.html ./
COPY src ./src

# `npm run build` runs tsc first, so a type error fails the image build rather
# than shipping a bundle that only breaks at runtime.
RUN npm run build


FROM nginx:1.27-alpine AS runtime

COPY --from=build /build/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# nginx's own image drops to an unprivileged user for worker processes but
# needs root to bind and to write its pid; 8080 rather than 80 keeps this
# consistent with the non-root ports the rest of the stack uses.
EXPOSE 8080

# 127.0.0.1, not localhost. The alpine image's /etc/hosts maps localhost to
# both 127.0.0.1 and ::1, wget tries the IPv6 address first, and nginx's
# `listen 8080;` binds IPv4 only — so the probe was answering "connection
# refused" against a server that was serving every real request perfectly.
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:8080/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
