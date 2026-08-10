# Deploying FinSight

The whole stack is already the deploy artifact: `docker-compose.yml` builds
the Python image for `api`, a separate static bundle behind nginx for `web`,
and Qdrant is the third service. This
walks through putting that on a host reachable from the public internet — a
VPS, a spare machine, whatever you have — not a specific cloud platform's
one-click button. Nothing here needs code changes; it needs a machine,
DNS (optional), and a TLS terminator.

**This document is not applied automatically.** No deployment happens by
running anything in this repo — it is a guide for you to follow on a host
you control.

---

## Before you start

- **A host with Docker and Docker Compose installed.** Any VPS with 2GB+ RAM
  works — fastembed's ONNX model and Qdrant both want some headroom. This is
  the one real cost: even the cheapest VPS is a few dollars a month, running
  continuously, for as long as the deploy stays up.
- **API keys**, provisioned per [`docs/api_keys.md`](api_keys.md) — everything
  required is free, but Vertex AI is billed per token once you're actually
  fielding public traffic instead of your own testing.
- **A domain, if you want a real URL.** Optional — an IP address with a port
  works too, just without TLS (browsers will warn on the UI's own network
  calls if you mix HTTP and HTTPS, so decide this before you have real users).

## What NOT to do

- **Do not point `QDRANT_URL` at the host's public IP.** Qdrant's HTTP API
  has no auth in this project's config — see `src/vectorstore/client.py`'s
  isolation assertion, which protects against a *misconfigured URL*, not
  against an *open one*. Keep Qdrant on the compose network only (the
  default — `api`'s `QDRANT_URL: http://qdrant:6333` already never leaves
  it) and never publish port 6333/6335 in the firewall step below.
- **Do not commit the `.env` you create on the server.** Same rule as local
  dev — it is gitignored for a reason, and a server is not a safer place
  for a leaked key than a laptop.
- **Do not commit the service-account key either.** If you follow step 2's
  server path, `~/finsight-sa.json` is a live Vertex credential — keep it
  outside the repo directory entirely, `chmod 600`, and revoke it with
  `gcloud iam service-accounts keys delete` if it is ever exposed.
- **Do not skip the firewall step.** Compose's default `ports:` mapping
  binds to every interface — without a firewall, `api` (8000) and `web`
  (3000) are reachable directly, bypassing whatever auth or TLS you put in
  front of them.

---

## 1. Get the code and secrets onto the host

```bash
git clone <your-fork-or-remote-url> finsight
cd finsight
cp .env.example .env
# fill in .env — see docs/api_keys.md
chmod 600 .env
```

## 2. Give the `api` container Google credentials

This is the step that is easy to miss, because the stack comes up *healthy*
without it and only fails when someone asks a question — the container
starts, `/health` returns 200, and the first `POST /research/query` returns
a 500 with `MissingCredentialError` in the logs.

The reason: the image runs as its own unprivileged user (uid 1000,
`finsight`), so `HOME` inside the container is `/home/finsight`. Vertex
auth resolves Application Default Credentials relative to *that* home, not
to the host user's `~/.config/gcloud`. `docker-compose.yml` therefore
bind-mounts the credential file into the container read-only; all you
choose is which file it mounts.

**On a machine with a browser** (your laptop, a desktop) — the default,
nothing to configure:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
```

Leave `GOOGLE_CREDENTIALS_FILE` blank in `.env` and compose picks up
`~/.config/gcloud/application_default_credentials.json` automatically.

**On a headless server** there is no browser to complete that login, so use
a service-account key instead:

```bash
SA="finsight-api@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com"

gcloud iam service-accounts create finsight-api --display-name="FinSight API"
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
    --member="serviceAccount:${SA}" --role="roles/aiplatform.user"
gcloud iam service-accounts keys create ~/finsight-sa.json --iam-account="$SA"
chmod 600 ~/finsight-sa.json
```

Then point `.env` at it — an absolute path, since compose resolves it on
the host:

```bash
GOOGLE_CREDENTIALS_FILE=/home/youruser/finsight-sa.json
```

`roles/aiplatform.user` is the whole grant. It does not need project
editor, and it should not have it — this key only ever calls Vertex.

If the file named here does not exist, `api` **refuses to start** with a
bind-mount error rather than starting and failing later. That is deliberate:
Docker's default behaviour would silently create the missing path on the
host as an empty directory, which permanently breaks any later
`gcloud auth application-default login`.

> Running `GEMINI_BACKEND=aistudio` instead? Delete the credential mount from
> `docker-compose.yml`'s `api` service — that backend authenticates with
> `GOOGLE_API_KEY` from `.env` and needs no ADC at all.

## 3. Bring up the stack

```bash
docker compose up -d --build
docker compose ps          # all three services should be healthy within ~30s
curl -sf http://localhost:8000/health
```

Health checks only prove the process is up. Confirm the credential path
too, which the health endpoint deliberately does not touch:

```bash
curl -sf -X POST http://localhost:8000/research/query \
    -H 'Content-Type: application/json' \
    -d '{"query":"What was Apple'\''s revenue in the most recent fiscal year?"}'
```

A 200 means Vertex authenticated. A 500 means step 2 is not done — check
`docker compose logs api` for `MissingCredentialError`.

This is exactly what [`docs/dedup_algorithm.md`](dedup_algorithm.md)'s and
this project's own CI-equivalent (`make lint && make test`) already assume
works, because it is the same `docker-compose.yml` used in dev — see
[`README.md`](../README.md#quick-start).

## 4. Put a reverse proxy in front — this is what makes it "public"

Neither `api` nor `web` terminate TLS themselves. **[Caddy](https://caddyserver.com/)**
is the least ceremony for a single-host deploy: it gets a Let's Encrypt
certificate automatically from just a domain name, no separate certbot step.

```
# /etc/caddy/Caddyfile
finsight.example.com {
    basic_auth {
        demo $2a$14$REPLACE_THIS_WITH_YOUR_OWN_HASH
    }
    reverse_proxy localhost:3000
}
```

`reverse_proxy` needs only that one line, because the `web` container's own
nginx already proxies `/api/*` through to `api:8000` on the compose network
(`frontend/nginx.conf`). The browser only ever sees one origin, so there is
no path split to get wrong and no CORS preflight in the picture at all. Point
`finsight.example.com`'s DNS A record at the host, install Caddy, and
`systemctl restart caddy`.

Generate the credential with `caddy hash-password`, which prompts for a
password and prints a bcrypt digest. Paste the digest verbatim — a Caddyfile
does not expand `$`, so `$2a$14$…` needs no escaping. Do not commit a real
hash to this repo; bcrypt is expensive to crack, not impossible, and this
repo is public.

(The directive was named `basicauth` before Caddy 2.8. That spelling still
works as a deprecated alias, so an older Caddyfile will not break, but new
ones should use `basic_auth`.)

**This is not an optional extra.** There is no authentication anywhere
inside the application — see §6. This
directive is therefore the *only* access control on the deploy, and what sits
behind it is not merely a read-only dashboard:

- `POST /api/research/query` and `POST /api/monitor/cycles` spend real Vertex
  quota on every call, with no per-day ceiling on Gemini in `DAILY_BUDGETS`.
- A monitoring cycle fetches from SEC EDGAR under the real contact address in
  `SEC_USER_AGENT`, so someone else's traffic is attributed to you.
- `POST /api/monitor/cycles/{id}/resume` decides a paused HIGH-severity
  alert. Unauthenticated, "pauses for a human decision" means *any* human.

Put it in place on the first `systemctl reload caddy`, not after the link has
circulated. Three things it deliberately does not break:

- **The container healthcheck.** `docker-compose.yml` probes
  `http://localhost:8000/health` from inside the compose network, which never
  passes through Caddy. Guarding the public edge leaves `service_healthy`
  intact, and with it the `web` container's `depends_on`.
- **Certificate renewal.** Caddy answers the ACME challenge ahead of site
  routes, so an authenticated site still renews unattended.
- **The dashboard's own calls.** The browser caches the credential per origin
  once the document itself has authenticated, so every later `fetch` in
  `frontend/src/api/client.ts` carries it — the SSE stream included.

**Do not split the dashboard and API across two subdomains.** The bundle
resolves its API base at BUILD time (`VITE_API_URL`, see
`frontend/Dockerfile`), so an `API_URL` in `.env` — which is what the
Streamlit dashboard reads at container start — has no effect on it. If you
genuinely need `api.example.com`, rebuild the image with
`--build-arg`/`VITE_API_URL=https://api.example.com` and narrow
`allow_origins` in `src/api/main.py` to match, rather than expecting the
runtime env var to do it.

Any reverse proxy works the same way (nginx + certbot, Traefik, a cloud
load balancer). Two requirements beyond "TLS terminates somewhere in front
of 3000": it must **not buffer responses**, and its read timeout must
outlast a research run. `POST /api/research/query/stream` holds a connection
open for 30–60 seconds and delivers progress frames throughout; a proxy that
buffers turns that back into a spinner, and one with a 30-second timeout cuts
answers off mid-run. Caddy streams by default and its default timeouts are
unlimited, so the one-liner above is already correct — nginx in front would
need `proxy_buffering off;` and a raised `proxy_read_timeout`, exactly as
`frontend/nginx.conf` sets for the same reason.

## 5. Firewall

Only the reverse proxy's ports should be reachable from the internet.

```bash
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp    # or your actual SSH port
ufw enable
```

`3000`, `8000`, `6333`, `6335`, `6336` stay firewalled — Caddy reaches them
over `localhost`, not the public interface, so they never need to be open.
(`8501` too, if you also brought up the legacy Streamlit dashboard with
`--profile legacy`.)

## 6. Before this stops being "a demo with a URL"

The project's own README already says this API is CORS-wide-open and
single-user with no credentials to steal — see `src/api/main.py`. That is a
deliberate, documented tradeoff for a portfolio deploy, not an oversight,
but it means:

- **The application itself authenticates nobody.** Every route in `src/api/`
  is anonymous; there is no middleware, no dependency guard, and no API key
  check. §4's `basic_auth` block is the whole of the access control, and it
  is doing more work than it looks like — re-read §4 before removing it.
- **Basic auth is a door, not an identity.** One shared credential
  authenticates everybody who has it, which leaves two gaps it cannot close.
  There is still no per-caller rate limiting, so an authorised visitor can
  spend the quota as freely as an attacker could have. And an approval
  through `POST /monitor/cycles/{id}/resume` still cannot be attributed to a
  person, which is the one property an audit trail most wants. Closing
  either means auth *inside* the app — OIDC against Google with an email
  allowlist is the small version, since `google-auth` is already an installed
  dependency and no user table would be needed.
- **`allow_origins=["*"]`** in `src/api/main.py` should be narrowed once
  there is one real origin to narrow it to. Note the single-origin setup
  above means the browser never sends a cross-origin request at all, so
  tightening this costs nothing — it only closes the door on someone else's
  page scripting your API from a different origin.
- This project explicitly does not do multi-user auth, rate limiting per
  caller, or Kubernetes — see "Not doing" in the README. A public link
  behind basic auth is the intended ceiling, not a first step toward more.

## 7. Updating

```bash
git pull
docker compose up -d --build
```

Compose rebuilds only what changed; `finsight-data` and `finsight-qdrant`
are named volumes, so the alert history, watchlist, and dedup index all
survive a redeploy. Back them up if that history matters:

```bash
docker run --rm -v finsight_finsight-data:/data -v "$(pwd)":/backup alpine \
    tar czf /backup/finsight-data-backup.tar.gz -C /data .
```

## 8. Tearing down

```bash
docker compose down          # stop and remove containers, KEEP volumes
docker compose down -v       # also delete the alert history and dedup index
```
