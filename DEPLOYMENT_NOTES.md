# FinSight — Live Deployment Notes

Untracked on purpose (see `.gitignore`) — this is a record of one specific
deploy's actual state, not documentation for others to follow. For the
general how-to, see [`docs/deploy.md`](docs/deploy.md).

## Where it lives

| | |
|---|---|
| URL | https://finsight.thisenekanayake.me |
| GCP project | `finsight-504309` |
| VM name | `finsight` |
| Zone | `asia-south1-c` |
| Machine type | `e2-medium` |
| Static IP | `34.93.151.52` (reserved as `finsight-ip`, region `asia-south1`) |
| SSH | `gcloud compute ssh finsight --zone=asia-south1-c --project=finsight-504309` |
| Repo on VM | `~/FinSight` (cloned from `git@github.com:Thisen-Ekanayake/FinSight.git` / `https://github.com/Thisen-Ekanayake/FinSight.git`) |
| Deployed commit | `14a78d7` (path routing; PR #3 shell_scripts came along in the same pull) as of 2026-08-11 |
| Sign-in | Google, two tiers (free / unlimited) — see "User auth" below |

## Auth: no key file, metadata-server ADC

`GOOGLE_CREDENTIALS_FILE`-based auth from `docs/deploy.md` §2 does **not**
apply to this deploy. Project `finsight-504309` enforces the
`constraints/iam.disableServiceAccountKeyCreation` org policy — `gcloud iam
service-accounts keys create` fails outright, for the owner account too, not
just restricted callers.

Instead, the VM authenticates via its **attached service account** through
the GCE metadata server — no key file exists anywhere:

- Service account: `finsight-api@finsight-504309.iam.gserviceaccount.com`
- IAM role: `roles/aiplatform.user` only (not editor)
- Attached to the VM with `--scopes=cloud-platform` (the default compute SA's
  scopes do NOT include this — had to `gcloud compute instances
  set-service-account` while the VM was stopped, then restart)
- Verify from the VM: `gcloud auth list` should show `finsight-api@...` as
  active, and `curl -H "Metadata-Flavor: Google"
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/scopes`
  should include `cloud-platform`

**`docker-compose.yml` on the VM is hand-edited**, diverging from git: the
credential bind-mount block (the one targeting
`/home/finsight/.config/gcloud/application_default_credentials.json`) is
deleted, since there is no file to mount — `google.auth.default()` inside
the container reaches the metadata server directly (confirmed: containers
on this VM can resolve and reach `metadata.google.internal` on the default
bridge network with no extra config).

**This means every `git pull` on the VM will conflict on
`docker-compose.yml`.** After pulling, reapply the deletion — the block to
remove starts at the comment `# Vertex auth. The image runs as uid 1000` and
ends at the line `create_host_path: false`. (A `docker-compose.override.yml`
using compose's `!override` merge tag would avoid this divergence, but
wasn't set up — todo if this gets tedious.)

## Code changes this deploy required (already merged to `main`)

The GCE-metadata-server auth path above surfaced a real gap, not a
deploy-specific hack — fixed upstream, not worked around locally:

- `d69d6d5` — `validate_llm_credentials()` now calls `google.auth.default()`
  instead of only checking for an explicit `GOOGLE_APPLICATION_CREDENTIALS`
  env var or the interactive-login ADC file. That means it now also accepts
  GCE/Cloud Run/GKE attached-service-account credentials.
- `8779290` — a stale test mock (`Path.is_file`) left over from the old
  implementation, which broke CI without breaking local runs.
- `faf484d` — pinned `streamlit==1.60.0` in `requirements.txt` (was
  unpinned; CI's fresh install picked up 1.61.1, which changed
  `AppTest.from_file`'s relative-path resolution and broke all 15
  `test_ui_pages.py` tests in CI only).

## Network

- Firewall rule `finsight-allow-web`: tcp:80,443 from `0.0.0.0/0`, scoped to
  VM tag `finsight-web` (the VM has this tag)
- Port 22 already covered by the project's pre-existing `default-allow-ssh`
- Nothing else is exposed — `8000` (api), `3000` (web, though Caddy proxies
  to it over `localhost`), `6335`/`6336` (qdrant) all stay off the GCP
  firewall entirely, so no host-level `ufw` was needed on top

## Reverse proxy / TLS

- Caddy 2.11.4, installed via the official apt repo
- `/etc/caddy/Caddyfile`:
  ```
  finsight.thisenekanayake.me {
      reverse_proxy localhost:3000
  }
  ```
- Cert: Let's Encrypt, auto-provisioned by Caddy on first reload — no
  certbot, no manual renewal setup needed
- DNS: A record for `finsight.thisenekanayake.me` → `34.93.151.52`
  (already in place before Caddy was configured — set at whatever registrar
  hosts `thisenekanayake.me`, not GCP Cloud DNS)

## Verified working (as of this deploy)

- `curl https://finsight.thisenekanayake.me/` → UI loads
- `curl https://finsight.thisenekanayake.me/api/health` → 200,
  `qdrant_detail: connected`
- `POST /api/research/query` through the public HTTPS endpoint → real
  Vertex-authenticated answer with citations, verification passed

## User auth: Google sign-in, live since 2026-08-11

Not to be confused with the Vertex ADC section above — that is how the API
authenticates to *Google*, this is how *people* authenticate to the API.

- OAuth client (project `finsight-504309`), type Web application:
  `264766702351-7lje65ehspka0upgcrrpojaihefolfq0.apps.googleusercontent.com`
- Authorised JavaScript origin: `https://finsight.thisenekanayake.me`
- No authorised redirect URI, deliberately — GIS returns the token to the page
- Consent screen: External, **Testing** mode, non-sensitive scopes only
  (`openid`/`email`/`profile`), so no Google verification review applies
- `.env` on the VM now carries `AUTH_ENABLED=true`, `GOOGLE_OAUTH_CLIENT_ID`
  and `AUTH_ALLOWED_EMAILS=thisenekanayake6@gmail.com`

### Free tier (added 2026-08-14) — ONE BLOCKER BEFORE IT WORKS PUBLICLY

`AUTH_ALLOWED_EMAILS` is now the *unlimited tier*, not an admission list: any
verified Google account may sign in and gets `FREE_QUERY_LIMIT` (default 5)
lifetime research queries, then a 402 pointing at `CONTACT_URL`.

**The consent screen above is in Testing mode, and that caps this at 100
manually-added test users.** In Testing mode Google refuses the sign-in itself
for anyone not listed under *Audience → Test users* — the request never reaches
FinSight, so nothing in this repo can admit them. To actually open the free
tier: Google Cloud console → *APIs & Services* → *OAuth consent screen* →
**Publish app**. Scopes are `openid`/`email`/`profile`, all non-sensitive, so
publishing needs no Google verification review and takes effect immediately.

The VM's `.env` also needs the two new keys (both have working defaults, so
this is optional):

```bash
FREE_QUERY_LIMIT=5
CONTACT_URL=https://github.com/Thisen-Ekanayake/FinSight
```

No migration step: `free_query_quotas` is a new table and `init_db()`'s
`create_all` picks it up on the next boot.

### Per-user history (added 2026-08-14) — no manual step

`research_runs` gained `subject` and `owner_email`. `create_all` cannot add a
column to an existing table, so `init_db()` now also runs
`_sync_additive_schema`, which diffs the live schema against the models and
issues `ALTER TABLE ... ADD COLUMN` plus any missing index. It is additive
only — never drops, renames or retypes — so reverting the application code
leaves a database the old code still runs against.

**On the VM this happens by itself on the next boot.** The existing rows
backfill to `subject = ''`, which means *unattributable*: they stay visible to
`AUTH_ALLOWED_EMAILS` accounts and to nobody else. Nothing is deleted. CLI runs
record the same empty subject, for the same reason — there is no identity at a
shell prompt.

To confirm afterwards:

```bash
docker compose exec api python -c "
from src.persistence.db import get_engine
from sqlalchemy import inspect, text
print([c['name'] for c in inspect(get_engine()).get_columns('research_runs')])
with get_engine().connect() as c:
    print(c.execute(text(
        'SELECT subject, count(*) FROM research_runs GROUP BY 1')).fetchall())
"
```

Verified live after the deploy: `/api/health` and `/api/auth/config` 200;
`/api/admin/*`, `/api/research/*`, `/api/monitor/*`, `/api/watchlist` all 401;
`/api/docs` and `/api/openapi.json` 404 (they are switched off whenever auth is
on — FastAPI registers them outside every router, so they cannot be guarded).

**Caddy `basic_auth` was never applied and is not needed now.** `docs/deploy.md`
§4 documents it for a deploy without app-level auth; this one has it.

## The `git pull` trap on this VM, concretely

The hand-edited `docker-compose.yml` divergence bites on every pull, and the
obvious way to reapply it is wrong. The block to delete opens at the comment
`# Vertex auth. The image runs as uid 1000` and closes at the **directive**
`create_host_path: false` — but one of the block's own comment lines contains
that exact string, so a naive first-match search stops early, deletes only the
comments, and leaves the bind mount in place. The container then refuses to
start against an ADC file that does not exist on this VM.

Match on the directive (a line whose stripped form *equals*
`create_host_path: false`, i.e. no leading `#`), then confirm with:

```bash
docker compose config | grep -i application_default_credentials   # must be empty
```

The trap only springs when an incoming commit actually touches
`docker-compose.yml`. It did **not** on the `c459be6 → 14a78d7` pull — neither
the routing change nor PR #3 went near that file, so git fast-forwarded and
carried the local edit through untouched. Worth checking with
`git diff --stat <deployed>..<target> -- docker-compose.yml` *before* pulling:
an empty result means a plain `git pull --ff-only` is safe and nothing needs
reapplying afterwards.

## Open items / not yet done

- **No global rate limiting.** The free tier caps queries per account, but a
  Google account is free to create, and the unlimited tier is uncapped by
  design. `DAILY_BUDGETS` still sets no Gemini ceiling, so there is nothing
  bounding total Vertex spend. A global daily cap reusing `ApiBudget` is the
  real fix before this is advertised anywhere.
- ~~**Research history is shared across accounts.**~~ **Fixed 2026-08-14.**
  `research_runs` now carries `subject` (Google's `sub`), `/research/runs`
  lists only the caller's, and `/research/threads/{id}` refuses anyone else's
  with a 404 worded identically to an unknown thread. Supplying another
  account's `thread_id` on a query is also refused — that was a write
  primitive against another account, and it is checked *before* the query
  meter so a refusal costs nothing.
- **Approvals are authenticated but not attributed.** `require_user` knows who
  resumed a cycle; `AlertRecord` has no `approved_by` column yet. **No longer
  blocked** — the migration decision that held this up was settled on
  2026-08-14 by `_sync_additive_schema` in `src/persistence/db.py`, which adds
  columns and indexes to live tables. `approved_by` is now a two-line model
  change plus a value at the call site.
- `ENVIRONMENT=development` in the VM's `.env` on what is a production host.
  Harmless today (it only gates a warning that auth being on makes moot), but
  it is mislabelled.
- `docker-compose.yml`'s divergence on the VM is a standing maintenance cost on
  every future `git pull` there — see the trap section above.
- Backups: `finsight-data` and `finsight-qdrant` are named Docker volumes on
  the VM only — no off-VM backup configured yet. See `docs/deploy.md` §8 for
  the tar-based backup command if this starts mattering.
- `tests/test_ingest.py::...::test_ingest_then_reingest_does_not_duplicate` is
  flaky (asserts on Qdrant's approximate `points_count` after a `wait=False`
  bulk upsert). Pre-existing, unrelated to auth; fix is to assert on
  `client.count(..., exact=True)`.
