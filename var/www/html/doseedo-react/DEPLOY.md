# Deploy Guide — `doseedo-react`

This document is the source of truth for getting frontend changes into
production at `doseedo.com`. **Read at least the Architecture section
before deploying for the first time** — the layout is non-obvious and
the wrong mental model leads to "I deployed and nothing changed"
debugging spirals.

---

## TL;DR

```bash
cd var/www/html/doseedo-react
./deploy.sh
```

That builds the React app, syncs `build/` to the GCS bucket, rebuilds
the Cloud Run nginx image, invalidates GCP CDN, purges Cloudflare, and
verifies that the bundle hash matches what production is serving.
Takes ~3–5 minutes including verification.

If you only changed React code (no `nginx.conf`, no `Dockerfile`, no
`login.html`):

```bash
./deploy.sh --skip-cloud-run
```

That trims off the slowest step (~90s of Cloud Build).

---

## Architecture (read this first)

`doseedo.com` is **not** served by a single thing. It's three separate
artifacts that the Google Cloud Load Balancer URL map (`bassify`)
stitches together by URL path. The URL map config is checked into
this repo at `ops/bassify-urlmap.yaml`.

```
                    ┌─────────────────────────┐
   doseedo.com  ──► │  Cloudflare (CDN edge)  │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  GCLB url-map: bassify  │
                    │  pathMatcher: frontend- │
                    │             split       │
                    └─────┬──────────┬────────┘
                          │          │
              default     │          │  /home, /home/, /api/encode-*,
              (everything │          │  /separate-stems/*, /api/generate-
              not matched │          │  stemphonic, /_chat/ws, ~10 more
              below)      │          │  (see ops/bassify-urlmap.yaml
                          │          │   pathRules)
                          ▼          ▼
            ┌─────────────────┐  ┌──────────────────────────┐
            │  GCS bucket     │  │  Cloud Run service       │
            │  doseedo-       │  │  doseedo-frontend        │
            │  frontend-      │  │  (region us-central1)    │
            │  static         │  │                          │
            │                 │  │  Built from this dir's   │
            │  Holds CRA's    │  │  Dockerfile. Runs nginx  │
            │  build/ output  │  │  with this dir's         │
            │  (index.html,   │  │  nginx.conf and ALSO     │
            │  static/js,     │  │  bundles login.html +    │
            │  static/css,    │  │  the same build/ dir.    │
            │  asset-         │  │                          │
            │  manifest.json, │  │  Used for the Framer-    │
            │  public/**)     │  │  proxied /home route +   │
            │                 │  │  ~12 API/inference paths │
            └─────────────────┘  └──────────────────────────┘
```

### What that means in practice

- **Almost every URL on `doseedo.com`** (`/dashboard`, `/projects`,
  `/plugins`, `/profile`, `/help`, `/feedback`, `/plans`, `/login`,
  the SPA bundles, the `public/` assets) resolves to the **GCS bucket**.
  When you edit a React component, the change has to land in
  `gs://doseedo-frontend-static`.

- **The Cloud Run service is narrow**. It only serves the routes the
  URL map sends to it: `/home` (Framer reverse-proxy), a handful of
  AI/inference `/api/*` endpoints, `/separate-stems/*`, etc. See
  `ops/bassify-urlmap.yaml` for the exact list (`pathRules` under
  `frontend-split`).

- **Both artifacts share the same `build/` directory**. The Dockerfile
  copies `build/` into the nginx image (line 10) AND `deploy.sh`
  rsyncs `build/` to the bucket. So a single `npm run build` produces
  the assets that go to both places, and you almost always want to
  push to both at once so they stay in lockstep.

- **`nginx.conf` lives in the Cloud Run service only**. None of its
  `location` blocks or `rewrite` rules apply to the bucket-served
  paths. The `rewrite ^/plans\.html$ /plans permanent;` style rules
  in this repo's `nginx.conf` only fire for requests that the URL
  map routed to Cloud Run — i.e. `/home` and the API list. They do
  **not** fix `doseedo.com/plans.html` because that path goes to the
  bucket, where it 404s and the URL map's `customErrorResponsePolicy`
  rewrites the 404 to `/index.html` (SPA fallback). If you need a
  legacy URL alias for a bucket-served path, the right place is
  either the URL map (`pathRules`) or a redirect HTML file uploaded
  into the bucket.

- **Cloudflare sits in front of GCP**. Its cache is independent of GCP
  Cloud CDN — invalidating the GCLB url-map cache does **not** purge
  Cloudflare. If you skip the CF purge step, users will see stale
  CSS/JS until Cloudflare's TTL expires (could be hours). The token
  to purge lives at `/scratch/cache/secrets/cloudflare-token`
  (chmod 600).

### Bucket vs backendBucket name gotcha

The URL map references `backendBuckets/doseedo-frontend-bucket` but
the actual GCS bucket name is `doseedo-frontend-static`. The
backendBucket is a GCLB resource that wraps the GCS bucket, and the
two names are independent. If you `gsutil ls gs://doseedo-frontend-bucket`
you'll get nothing — that's not the real bucket. The deploy script
uses the right name. **Don't trust the URL map's resource name as a
GCS path.**

---

## What needs deploying when?

Most changes go to both the bucket AND the Cloud Run image, because
the React build is bundled into both. Use this table to know which
flags to pass to `deploy.sh`:

| Change | Bucket sync (step 1+2) | Cloud Run rebuild (step 3) | URL map import |
|---|---|---|---|
| Edit React component (`.js`/`.jsx`/`.module.css`) | ✅ | ✅ (it's bundled into the image too) | ❌ |
| Edit `original-style5.css`, `Home.module.css`, etc. | ✅ | ✅ | ❌ |
| Edit `nginx.conf` | ❌ (n/a) | ✅ | ❌ |
| Edit `login.html` | ❌ (it's only baked into the Cloud Run image, not the bucket) | ✅ | ❌ |
| Edit `Dockerfile` | ❌ | ✅ | ❌ |
| Add a new SPA route in `App.js` currentView | ✅ | ✅ | ❌ (the URL map's 404 → /index.html fallback handles it) |
| Add a new `/api/*` endpoint that should hit Cloud Run | ❌ | ❌ | ✅ (add to `ops/bassify-urlmap.yaml`) |
| Add a new path that should redirect | depends — see below | depends | ✅ if it's a bucket path |
| Edit `ops/bassify-urlmap.yaml` | ❌ | ❌ | ✅ (`gcloud compute url-maps import bassify --source ops/bassify-urlmap.yaml`) |

In practice you almost always run the full `./deploy.sh`. The
`--skip-cloud-run` flag is only worth it when you're iterating fast on
React-only changes and want to save the ~90 seconds of Cloud Build per
deploy. Even then, if you forget that nginx.conf edits don't ship with
the bucket sync, you'll waste more time debugging "why isn't my rewrite
working" than you saved.

---

## Step-by-step: what `deploy.sh` actually does

### 0. `npm run build`
Standard CRA production build. Outputs to `./build/`. Skipped if
`--skip-build` is passed (use that when you've already built and just
want to re-deploy without recompiling). The script aborts if
`--skip-build` is set but `build/` doesn't exist.

### 1. `gsutil -m rsync -r -c -d build/ gs://doseedo-frontend-static/`

`-m` parallel, `-r` recursive, `-c` checksum-compare (not size-compare —
critical, see step 2), `-d` delete files in the bucket that aren't in
`build/` (so old hashed bundles get cleaned up).

### 2. Force-upload `index.html` and `asset-manifest.json` with `Cache-Control: no-store`

Two reasons this exists:
1. **Cache-Control**: `index.html` is the entry point — it references
   the latest hashed bundles. If Cloudflare/CDN caches it, users with
   warm caches will request the OLD `main.[hash].js` which no longer
   exists in the bucket → 404. Setting `no-store, max-age=0` tells
   the CDN never to cache it.
2. **Force upload**: even with `-c` checksum mode, `gsutil rsync` has
   historically had edge cases where it skips files. `index.html` and
   `asset-manifest.json` are the two files where being out of date is
   catastrophic, so we just always re-upload them with an explicit
   `cp`.

### 3. `gcloud run deploy doseedo-frontend --source . --region us-central1 --quiet`

Cloud Build picks up the `Dockerfile`, builds a new image, pushes to
Artifact Registry, and Cloud Run promotes it as the new revision with
100% traffic. Takes ~60–120s.

This is the step that:
- Rebuilds the nginx config (any `nginx.conf` change goes live here)
- Re-bundles `login.html` into the image
- Re-bakes the React `build/` into the image (so the Cloud Run service
  can serve the SPA at any of its routed paths, in addition to its
  Framer proxy + API duties)

### 4. `gcloud compute url-maps invalidate-cdn-cache bassify --path "/*"`

Tells GCP Cloud CDN to drop everything cached against the `bassify`
url-map. Async — returns immediately, propagation takes a few seconds.

### 5. Cloudflare purge

`POST https://api.cloudflare.com/client/v4/zones/{ZONE}/purge_cache`
with `{"purge_everything": true}`. Token is loaded from
`/scratch/cache/secrets/cloudflare-token` (chmod 600). Skipped with a
warning if the file doesn't exist — the deploy still completes but
edge caches will keep serving stale assets until their TTL expires.

### 6. Verification

Sleeps 25s for CDN propagation, then:
```bash
EXPECTED=$(grep -oE 'main\.[a-f0-9]+\.js' build/index.html | head -1)
LIVE=$(curl -s "https://doseedo.com/?cb=$RANDOM" | grep -oE 'main\.[a-f0-9]+\.js' | head -1)
```
If `$EXPECTED == $LIVE`, the new bundle is being served and the
deploy is verified. If they differ, the script exits 3 with a hint
to retry the verification curl.

---

## Manual verification commands

If `deploy.sh` exits 3 (bundle mismatch) or you want to sanity-check
later:

```bash
# What does build/index.html reference?
grep -oE 'main\.[a-f0-9]+\.js' build/index.html | head -1
grep -oE 'main\.[a-f0-9]+\.css' build/index.html | head -1

# What is doseedo.com actually serving?
curl -s "https://doseedo.com/?cb=$RANDOM" | grep -oE 'main\.[a-f0-9]+\.(js|css)'

# Bypass Cloudflare entirely and hit the GCS bucket directly
gsutil cat gs://doseedo-frontend-static/index.html | grep -oE 'main\.[a-f0-9]+\.(js|css)'

# Bypass both Cloudflare and GCP CDN and hit the bucket origin
curl -s "https://storage.googleapis.com/doseedo-frontend-static/index.html" \
  | grep -oE 'main\.[a-f0-9]+\.(js|css)'

# What does the Cloud Run service serve at /home?
curl -sI "https://doseedo.com/home" | head -5
```

If `gsutil cat` shows the new hash but `curl doseedo.com` shows the
old one, the issue is purely CDN cache (purge again, wait longer, or
hard-refresh your browser). If `gsutil cat` shows the old hash, the
bucket sync didn't happen — check `deploy.sh` output for errors and
re-run.

---

## Troubleshooting

### "I deployed but `doseedo.com/dashboard` still looks the same"

In order of likelihood:

1. **Browser cache.** Cmd+Shift+R / Ctrl+Shift+R (hard refresh).
   Your browser is the most aggressive cache.
2. **Cloudflare cache didn't purge.** Re-run the CF purge step:
   ```bash
   CF_TOKEN=$(cat /scratch/cache/secrets/cloudflare-token)
   curl -X POST "https://api.cloudflare.com/client/v4/zones/d16198032affca1d6ef0978025c5b5a0/purge_cache" \
     -H "Authorization: Bearer $CF_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"purge_everything":true}'
   ```
3. **GCP CDN is still serving stale.** Re-run:
   ```bash
   gcloud compute url-maps invalidate-cdn-cache bassify --path "/*"
   ```
   Wait ~60s and re-verify.
4. **Bucket sync didn't actually push the new bundle.** Run the
   manual verification commands above and check `gsutil cat` output.
   If the bucket has the old hash, the rsync silently skipped or
   errored. Re-run `./deploy.sh --skip-cloud-run`.
5. **You forgot to commit, so `npm run build` is bundling old code.**
   `git status` to verify there are no uncommitted edits, then rerun.

### "Cloud Build failed during step 3"

Most common cause: lint warnings being promoted to errors under
`CI=true`. The Dockerfile runs `npm run build` inside Cloud Build,
and Cloud Build sets `CI=true` automatically. CRA promotes ESLint
warnings to errors when `CI=true`, and there's a long tail of
pre-existing `no-unused-vars` / `react-hooks/exhaustive-deps`
warnings in `components-demo/`. To bypass:

- Either fix the warnings (the right but tedious answer)
- Or add `ENV CI=false` to the `Dockerfile` build stage before
  `RUN npm run build`

The local `./deploy.sh` build (step 0) does NOT set `CI=true`, so it
succeeds with warnings. Cloud Build does. They diverge. This is a
known landmine.

### "build/ directory is missing"

`./deploy.sh` builds it for you. If you ran with `--skip-build` and
forgot to build first, the script aborts at step 0. Run `npm run build`
manually or drop the `--skip-build` flag.

### "gsutil: command not found" / "gcloud: command not found"

You need the gcloud SDK. `gsutil` ships as part of `gcloud`. Install
from https://cloud.google.com/sdk/docs/install or use the
Cloud-Run-aware container.

### "Permission denied on /scratch/cache/secrets/cloudflare-token"

Either the file doesn't exist, isn't readable by your user, or has the
wrong permissions. The script tolerates a missing token (skips step 5
with a warning) but warns that edge caches will keep serving stale
assets. To create the token: provision a new Cloudflare API token
with `Zone.Cache Purge` permission for the doseedo zone, save it to
`/scratch/cache/secrets/cloudflare-token`, and `chmod 600` it.

### "URL map import warns about a stale fingerprint"

Expected — the YAML in `ops/bassify-urlmap.yaml` was exported at a
specific fingerprint and the live url-map has moved on. `gcloud
compute url-maps import bassify --source ops/bassify-urlmap.yaml
--quiet` will succeed despite the warning. See `ops/README.md` for
the URL map runbook.

### "I added a new route to App.js but the URL 404s in production"

The url-map's `customErrorResponsePolicy` rewrites bucket 404s to
`/index.html`, so any unknown path in the bucket falls through to the
SPA, which then dispatches via React Router based on `pathname`. So
new SPA routes work without any url-map changes — but only if the
bundle was actually rebuilt and uploaded. If your route 404s in
production:

1. Verify it works locally (`npm start`, navigate to the route)
2. Check `gsutil cat gs://doseedo-frontend-static/index.html` —
   does the bundle hash match `build/index.html`?
3. If not, the deploy didn't push your changes. Re-run.

### "I edited nginx.conf but the rewrite isn't firing"

Two possibilities:
1. **You skipped step 3**. `nginx.conf` only ships via the Cloud
   Run image — `--skip-cloud-run` would have left the old config in
   place. Re-run `./deploy.sh` without that flag.
2. **The path doesn't route to Cloud Run.** Check
   `ops/bassify-urlmap.yaml` — if the path you're trying to rewrite
   isn't in the `pathRules` list pointing to `frontend-cloudrun-backend`,
   then requests for that path go to the GCS bucket and never hit
   nginx. To fix: either add the path to the url-map (and re-import),
   or stop trying to use nginx rewrites and instead handle the
   redirect at a different layer (a meta-refresh HTML in the bucket,
   or a url-map pathRule).

---

## Files involved in deploy

| File | Purpose |
|---|---|
| `deploy.sh` | The deploy script. Bring-your-own-tools (gcloud, gsutil, curl, python3). |
| `Dockerfile` | Builds the Cloud Run nginx image — `node:18-alpine` build stage runs `npm install` + `npm run build`, then copies `build/` + `login.html` + `nginx.conf` into a `nginx:alpine` runtime image. |
| `nginx.conf` | The nginx config baked into the Cloud Run image. **Only applies to paths the url-map routes to Cloud Run** (see Architecture). |
| `login.html` | Standalone sign-in page. Gets `COPY`'d into the nginx image at build time. NOT in the GCS bucket. |
| `package.json` / `package-lock.json` | npm dependencies. `package-lock.json` is critical for reproducible builds — make sure it's not gitignored (the project's `.gitignore` has explicit `!doseedo-react/package-lock.json` negation). |
| `public/index.html` | CRA template. The pre-React auth gate lives here — it inline-redirects unauthenticated visitors from `/` to `/home` before the SPA bundle parses. |
| `src/App.js` | The route definitions (`currentView` switch). Add new SPA routes here. |
| `../../../ops/bassify-urlmap.yaml` | URL map — the routing layer that decides bucket vs Cloud Run vs other backends. See `ops/README.md` for its own runbook. |

---

## History / why this doc exists

Before this file landed, the only deploy script was at
`/home/arlo/do2/deploy_studio.sh` outside the repo, with a hardcoded
`REPO=/scratch/do-repo/...` path that pointed to a directory that
doesn't exist on this machine. Future agents had no way to find it
without grep-ing the entire filesystem, and even if they did, the
script wouldn't run as-is.

This in-repo `deploy.sh` is functionally equivalent to that script
but uses `$(dirname "${BASH_SOURCE[0]}")` to resolve its own location,
so it works from any clone of the repo regardless of where the user's
home directory lives. The original `/home/arlo/do2/deploy_studio.sh`
is still on disk and unmodified — it's just no longer the canonical
deploy path.

If you find a discrepancy between this doc and what actually happens,
**this doc is wrong, the real system is right** — please update this
file as part of the same change that proves the discrepancy.
