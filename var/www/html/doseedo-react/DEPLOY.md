# Deploy Guide — `doseedo-react`

> ## ⚠️ DEPRECATED as of 2026-04-17 — production is on Vercel
>
> **`doseedo.com` is now served by Vercel**, not GCLB + GCS + Cloud Run.
> The `deploy.sh` script and everything about GCS buckets, Cloud Run
> services, and bundle-hash matching described below is no longer the
> production deploy path. **Do not run `deploy.sh` unless you're
> intentionally pushing to the rollback path (`legacy.doseedo.com`).**
>
> New deploy path:
> ```
> # Just edit + commit + push. Vercel auto-deploys main in ~2 min.
> cd /Users/hydroadmin/Downloads/Do/var/www/html/doseedo-react
> # ...make changes...
> git push
> ```
>
> Full reference: `CLAUDE.md` (FRONTEND DEPLOY section) or
> `doseedo-next/README.md`. The Next.js shell at `Do/doseedo-next/`
> wraps this CRA tree; `doseedo-next/sync.sh` keeps it in sync on each
> build. Edit here, push, done.
>
> ---
>
> ### What's still alive but unused
>
> - **GCS bucket** `gs://doseedo-frontend-static` — contains the last
>   CRA `build/` output. Reachable via `legacy.doseedo.com`.
> - **Cloud Run service** `doseedo-frontend` (us-central1,
>   `audiocraft-411005`) — old nginx image. Also on `legacy.doseedo.com`.
> - **GCLB url-map** `bassify` — still routes `api.doseedo.com` paths
>   to the auth / chatbot / stemphonic backends; also serves
>   `legacy.doseedo.com` for rollback.
>
> Safe to tear down GCS bucket + Cloud Run service after ~1 week of
> stable Vercel traffic. Confirm `api.doseedo.com` isn't still needed
> before touching the url-map.
>
> ### Emergency rollback from Vercel → legacy
>
> Fast path (DNS):
> ```bash
> source /Users/hydroadmin/Downloads/doseedo-desktop/.cloudflare
> # Point doseedo.com apex back at GCLB (34.160.103.40). Record IDs
> # are in Cloudflare — list via:
> curl -s -H "Authorization: Bearer $CF_API_TOKEN" \
>   "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/dns_records?name=doseedo.com&type=A"
> ```
> Or (easier) Vercel dashboard → Deployments → pick last known good →
> "Promote to production". That avoids DNS changes entirely.
>
> If you genuinely need to push a fix to the legacy GCLB/GCS path while
> it's still alive, the commands that used to live in this file are
> preserved below for history.
>
> ---

<details>
<summary>📜 Historical reference — GCS + Cloud Run deploy (DO NOT USE FOR PRODUCTION)</summary>

The doc below describes the pre-2026-04-17 architecture. Kept here for
reference in case the legacy path needs to be revived temporarily.

## TL;DR (legacy)

```bash
cd var/www/html/doseedo-react
./deploy.sh
```

That built the React app, synced `build/` to the GCS bucket, rebuilt
the Cloud Run nginx image, invalidated GCP CDN, purged Cloudflare, and
verified that the bundle hash matched what production was serving.

## Architecture (legacy)

`doseedo.com` was three separate artifacts stitched together by the
GCLB URL map (`bassify`):
- **GCS bucket** `doseedo-frontend-static` — static JS/CSS chunks + most paths
- **Cloud Run** `doseedo-frontend` — `/home` Framer proxy + some API paths
- **GCLB** `bassify` — routed by path

Both origins had to serve the **same `index.html` bundle hash** or users got
`Uncaught SyntaxError: Unexpected token '<'` (GCS 404 returned HTML, browser
parsed it as JS). This was the exact class of bug the Vercel migration
eliminated.

The Dockerfile copied the **pre-built** `build/` dir — running `npm run
build` inside Docker produced a different bundle hash and broke things.

### Debugging "old bundle still showing" (legacy)

```bash
curl -s "https://doseedo.com/?cb=$RANDOM" | grep -o 'main\.[a-f0-9]*\.js'
gcloud storage cat gs://doseedo-frontend-static/index.html | grep -o 'main\.[a-f0-9]*\.js'
curl -s "https://doseedo-frontend-1028528394180.us-central1.run.app/" | grep -o 'main\.[a-f0-9]*\.js'
```

All three had to match.

</details>
