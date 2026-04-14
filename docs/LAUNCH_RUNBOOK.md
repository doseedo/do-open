# Launch Runbook

Operational procedures for the doseedo production stack. Keep this
file terse — one section per procedure, each reproducible from a
cold start.

## Rotating `internal-secret`

The `INTERNAL_SECRET` value gates Modal's `before_request` → auth's
`/internal/generation/consume` call. It lives in two stores that
must hold the **byte-identical** value: GCP Secret Manager secret
`internal-secret` (consumed by the `doseedo-auth` Cloud Run service
via env binding `INTERNAL_SECRET=secretKeyRef:internal-secret/latest`)
and Modal secret `doseedo-gate` key `INTERNAL_SECRET` (consumed by
the `doseedo-stemphonic` Modal app's `before_request` hook).

### The trailing-newline foot-gun
`gcloud secrets versions add internal-secret --data-file=-` will
store whatever you pipe in, including any trailing newline. Cloud
Run injects the raw bytes into the env var, so the auth router
compares `"abcdef…\n"` to the header `"abcdef…"` (HTTP header clients
strip trailing whitespace) and returns `403 Forbidden` on every call
— which Modal then fail-closes into a blanket `503 Service Unavailable`
for users. Diagnosis signal in Modal logs:
`gate: unexpected HTTP 403 from auth-service — failing closed`.
Verify byte count with `gcloud secrets versions access latest
--secret=internal-secret | wc -c` — for a 64-char hex value, 64
is correct, 65 means you have a newline.

### Correct rotation procedure
Generate a fresh 64-char hex value, write it to a file without a
trailing newline, create a new Secret Manager version, update the
Modal secret to the same value, force-restart both services, then
disable prior SM versions so they can't be accidentally pinned:
```bash
# 1. Generate and store (no trailing newline — use printf, not echo)
NEW=$(openssl rand -hex 32)
printf %s "$NEW" > /tmp/internal-secret.clean
[ $(wc -c < /tmp/internal-secret.clean) -eq 64 ] || { echo "bad length"; exit 1; }
gcloud secrets versions add internal-secret --data-file=/tmp/internal-secret.clean

# 2. Sync Modal
modal secret create doseedo-gate --force \
  "AUTH_SERVICE_URL=https://doseedo-auth-wd7h2yezlq-uc.a.run.app" \
  "INTERNAL_SECRET=$NEW" \
  "DISABLE_ALL_GENERATION=false"

# 3. Force auth Cloud Run to pick up the new SM version
gcloud run services update doseedo-auth --region=us-central1 \
  --update-env-vars=_RESTART=$(date +%s)

# 4. Force Modal to pick up the new secret value
cd $REPO/modal && modal deploy modal_stemphonic.py

# 5. Smoke test
curl -s -o /dev/null -w "%{http_code}\n" \
  -X POST https://doseedo-auth-wd7h2yezlq-uc.a.run.app/internal/generation/consume \
  -H "X-Internal-Secret: $NEW" -H "Content-Type: application/json" -d '{}'
# Expected: 401 (missing JWT). 403 means the secret still doesn't match.

# 6. Disable prior SM versions after verification so no client can pin them
for v in $(gcloud secrets versions list internal-secret --format="value(name)" \
           | grep -v ^$(gcloud secrets versions list internal-secret --limit=1 --format="value(name)")$); do
  gcloud secrets versions disable --secret=internal-secret $v
done
rm /tmp/internal-secret.clean
```

## Related verification
After rotation, run the 11-shot smoke test against
`/api/generate-stemphonic` with a fresh free-tier user token. Expect
`200×10` then `429` on the 11th. Any `503` response means the secret
chain is broken — check Modal logs with `modal app logs
doseedo-stemphonic` for the `gate: unexpected HTTP 403 …` line. The
kill switch is a separate vector: `DISABLE_ALL_GENERATION=true` in
the same `doseedo-gate` secret returns `503 {"error":"Generation is
temporarily disabled"}` with no auth call made.
