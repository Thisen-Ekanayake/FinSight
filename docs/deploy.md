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

**This is not an optional extra** until §6 is in place. Out of the box the
application authenticates nobody — every route in `src/api/` is anonymous
while `AUTH_ENABLED` is false — so this directive is the only access control
the deploy has, and what sits behind it is not merely a read-only dashboard:

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

## 6. Require a Google sign-in

§4's `basic_auth` is a door, not an identity: one shared credential
authenticates everybody, so an approval through `POST
/monitor/cycles/{id}/resume` cannot be attributed to a person. This replaces
it with real accounts.

**Create the OAuth client** in the Google Cloud console, in the same project
the deploy already uses:

1. *APIs & Services → OAuth consent screen*. User type **External**. Fill in
   an app name and your support/developer email. The only scopes needed are
   `openid`, `email` and `profile`, all **non-sensitive** — so there is no
   Google verification review to sit through. Leaving the app in *Testing*
   with your own address as a test user is enough for a single-operator tool.
2. *APIs & Services → Credentials → Create credentials → OAuth client ID*.
   Application type **Web application**.
3. Under **Authorised JavaScript origins**, add the origin the dashboard is
   served from — `https://finsight.example.com`. Scheme and host must match
   exactly, no path, no trailing slash. Add `http://localhost:5173` too if you
   want `npm run dev` to sign in against this client.
4. No **Authorised redirect URI** is needed. Google Identity Services returns
   the token to the page; there is no server-side redirect leg in this flow.
5. Copy the client ID. It ends in `.apps.googleusercontent.com`, and it is
   **not a secret** — it appears in every OAuth flow, and `GET /auth/config`
   serves it to any browser that asks.

**Configure the deployment** in `.env`:

```bash
AUTH_ENABLED=true
GOOGLE_OAUTH_CLIENT_ID=1234567890-abc123.apps.googleusercontent.com
AUTH_ALLOWED_EMAILS=you@gmail.com,colleague@gmail.com
FREE_QUERY_LIMIT=5
CONTACT_URL=https://github.com/Thisen-Ekanayake/FinSight
```

Then `docker compose up -d --build`. The bundle needs rebuilding only because
it is a static image; the client ID itself is read at runtime from
`/auth/config`, so the same image works against any deployment.

Setting `AUTH_ENABLED=true` with no client ID **aborts startup** rather than
booting half-guarded: no token could ever be verified, while `/health` still
reported the service fine.

### What the two tiers mean

`AUTH_ALLOWED_EMAILS` is **not an admission list.** Any Google account with a
verified email may sign in. What the list decides is how much they get:

| | Free tier (anyone) | Unlimited tier (`AUTH_ALLOWED_EMAILS`) |
|---|---|---|
| `POST /research/query` and `/query/stream` | `FREE_QUERY_LIMIT` for the lifetime of the account, then **402** | unmetered |
| Dashboard, findings, run history, watchlist | yes | yes |
| `/monitor/*` — run or approve a cycle | **403** | yes |
| `/admin/budgets`, `/admin/config` | **403** | yes |

The monitor and admin routes are reserved because neither is metered and both
are expensive or revealing: a cycle spends LLM tokens per watched ticker, and
resuming one dispatches an alert. Opening sign-in to the world without that
split would hand both to anyone with a Google account.

`FREE_QUERY_LIMIT=0` restores the pre-free-tier behaviour — only the unlimited
tier can query — except the refusal is a 402 carrying `CONTACT_URL` instead of
a bare 403. An empty `AUTH_ALLOWED_EMAILS` now **warns instead of aborting**:
it is a legitimate configuration, it just means nobody is exempt and nobody can
reach the monitor.

The counter lives in `free_query_quotas`, keyed by Google's `sub` claim rather
than the email — an account that changes its address keeps one counter.
`research_runs.subject` uses the same key, so history follows an account
across an address change too.

### Schema changes on an existing database

`init_db()` runs `create_all()` for absent tables and then
`_sync_additive_schema`, which diffs the live schema against the models and
adds any missing **column or index** to a table that already exists. It is
additive only and never drops, renames or retypes, so rolling the application
back leaves a database the old code still runs against. Both of the above
land on a running deployment with no manual step and no downtime.

Anything destructive is deliberately out of scope — that wants a script and a
person watching. The helper refuses at startup, loudly, if a model declares a
column `ADD COLUMN` cannot create (a key, or `NOT NULL` with no server
default), so the failure lands in review rather than on the VM.

To grant someone more by hand:

```bash
docker compose exec api python -c "
from src.persistence.db import session_scope
from src.persistence.models import FreeQueryQuota
with session_scope() as s:
    for row in s.query(FreeQueryQuota).all():
        print(row.email, row.used)
"
```

**Check the boundary before trusting it:**

```bash
curl -so /dev/null -w '%{http_code}\n' https://finsight.example.com/api/health          # 200
curl -so /dev/null -w '%{http_code}\n' https://finsight.example.com/api/auth/config     # 200
curl -so /dev/null -w '%{http_code}\n' https://finsight.example.com/api/admin/budgets   # 401
curl -so /dev/null -w '%{http_code}\n' https://finsight.example.com/api/monitor/alerts  # 401
curl -so /dev/null -w '%{http_code}\n' https://finsight.example.com/api/auth/quota      # 401
```

`/health` answering 200 is not an oversight and must stay that way:
`docker-compose.yml` probes it to decide `service_healthy`, and the `web`
container's `depends_on` waits on that. Guard it and the dashboard never
starts.

**Once this works, remove §4's `basic_auth` block** and reload Caddy. It has
been superseded — leaving it on means signing in twice, through a browser
credential dialog that no longer protects anything the application does not.

**Three refusal codes, and they mean different things.** Keeping them distinct
is what stops the dashboard looping against a wall it cannot get past:

- **401** — no token, or an expired one. Signing in again fixes it, and the
  dashboard does exactly that.
- **402** — the free queries are spent. Carries a JSON body with `used`,
  `limit` and `contact_url`, which is what the "Contact sales" panel renders.
- **403** — verified, but this route is reserved for `AUTH_ALLOWED_EMAILS`.
  Retrying the sign-in cannot change it, so the dashboard does not try.

## 7. Before this stops being "a demo with a URL"

Assuming §6 is in place, what remains is narrower than it used to be — but it
is not nothing:

- **With `AUTH_ENABLED=false`, the application authenticates nobody.** Every
  route in `src/api/` is anonymous unless §6 is configured; there is no
  middleware and no API key check behind it. If you skipped §6, then §4's
  `basic_auth` block is the entire access control and it is doing far more
  work than it looks like — re-read §4 before removing it.
- **The free tier caps queries per account, not spend.** Google accounts are
  free and unlimited to create, so five lifetime queries costs a determined
  abuser about thirty seconds per five. There is still **no global ceiling**:
  `DAILY_BUDGETS` in `src/data/config.py` caps `fmp` and `alphavantage`, and
  there is no Gemini ceiling at all. Treat `FREE_QUERY_LIMIT` as a speed bump
  and a product decision, not as cost control. A global daily cap reusing the
  existing `ApiBudget` machinery is the real fix if this is ever public.
- **The unlimited tier has no roles inside it.** No per-user data isolation,
  no ownership — every allowlisted account can do everything, including
  approving another operator's paused alerts. Only `free_query_quotas` is
  per-account; no other table in `src/persistence/models.py` has an owner
  column.
- **Research history is per account, and that is the only isolation there is.**
  `GET /research/runs` lists only the caller's runs and
  `/research/threads/{id}` refuses anyone else's. Nothing else is scoped: the
  watchlist is global and mutable by any signed-in account, and monitoring
  alerts are shared across the unlimited tier. Runs recorded before this
  existed, and every run started from the CLI, count as unattributable and are
  visible to `AUTH_ALLOWED_EMAILS` only.
- **Approvals are authenticated but not yet recorded.** `require_user` knows
  who resumed a cycle; `AlertRecord` does not store it. Until it does, the
  audit trail proves a *permitted* human approved an alert, not *which* one.
- **Leave `CORS_ALLOW_ORIGINS` empty** unless the dashboard and API really are
  served from different origins. Empty means same-origin only, which is what
  the nginx setup above already is. Never set it to `"*"`: with a bearer token
  in the browser, that lets any page an operator visits drive this API from
  its own origin.
- **The sign-in session is the token's own lifetime** — about an hour, with no
  silent refresh. Expiry surfaces as a 401 and returns the dashboard to the
  sign-in screen, mid-research-run included. A backend-issued session cookie
  is the fix if that becomes tiresome; there isn't one today.
- This project does not do roles, per-user data isolation, or Kubernetes.
  See "Not doing" in the [README](../README.md#not-doing). A two-tier Google
  sign-in with a lifetime free-query counter is the intended ceiling, not a
  first step toward a user system with billing.

## 8. Updating

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

## 9. Tearing down

```bash
docker compose down          # stop and remove containers, KEEP volumes
docker compose down -v       # also delete the alert history and dedup index
```
