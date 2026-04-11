# ops/ — infrastructure snapshots & runbooks

This directory captures pieces of GCP state that aren't naturally tracked
by the application source tree, so they survive infra rebuilds / handoffs
and can be re-applied with `gcloud` without guessing what the config was.

## Files

### `bassify-urlmap.yaml`
Current exported state of the **`bassify` global URL map** (GCLB) as of
2026-04-11. This is what routes `doseedo.com/*` across the Cloud Run
services and the `doseedo-frontend-bucket` backend bucket.

The entry that's only in GCP (not derivable from application source) is
the first pathRule under `frontend-split`:

```yaml
- paths:
  - /home
  - /home/
  service: .../backendServices/frontend-cloudrun-backend
```

That rule is what makes `doseedo.com/home` transparently serve the
Framer marketing site via the `doseedo-frontend` Cloud Run service's
nginx.conf (`location = /home { proxy_pass framer }`). If this rule
ever goes missing, unauthenticated visitors land on a broken SPA
fallback at /home.

### `bassify-urlmap.pre-home-route.yaml`
Snapshot of the URL map from *before* the /home pathRule was added
(2026-04-11 18:45 UTC). Keep this around as a rollback target.

## Re-applying after an infra rebuild

```bash
gcloud compute url-maps import bassify \
  --source ops/bassify-urlmap.yaml \
  --quiet
gcloud compute url-maps invalidate-cdn-cache bassify --path "/*" --async
```

## Rolling back the /home pathRule

```bash
gcloud compute url-maps import bassify \
  --source ops/bassify-urlmap.pre-home-route.yaml \
  --quiet
gcloud compute url-maps invalidate-cdn-cache bassify --path "/*" --async
```

(The import will warn about a stale fingerprint on the yaml — that's
expected, `--quiet` is enough.)

## Related application-side files (already in git, listed here for ops reference)

| File | What it does |
|---|---|
| `var/www/html/doseedo-react/nginx.conf` | Cloud Run `doseedo-frontend` nginx config. `location = /home { proxy_pass https://heartwarming-friday-546447.framer.app/ }` is the actual proxy target — if you change the Framer URL, it's here. |
| `var/www/html/doseedo-react/public/index.html` | Inline pre-React cookie gate that redirects unauthenticated visitors to `/home` before the SPA bundle parses. Checks `username` cookie (contract lives in `src/services/authService.js`). |
| `var/www/html/doseedo-react/public/home/index.html` | Static bucket fallback for `/home`. Only served if the `bassify` pathRule is ever reverted; it meta-refreshes to https://do.doseedo.com/ as a last-resort safety net. |
| `var/www/html/doseedo-react/src/App.js` | SPA-side auth retry backstop that catches the race where the `username` cookie arrives after first paint. |
