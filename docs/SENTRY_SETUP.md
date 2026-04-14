# Sentry Setup

The code paths are all scaffolded and shipped in `4c9c32f`+ (frontend
`src/lib/sentry.js` + `telemetry.js`, backend `stemphonic_server.py`
init block, `deploy.sh` source-map step). They NO-OP gracefully
until the three bits below are in place. This doc is the punch list
to take them live.

## 1. Create the Sentry account + projects

- Sign up at https://sentry.io/signup/ with `arlo@doseedo.com`
- Create an org — pick a slug you're happy to see in URLs forever
  (e.g. `doseedo`)
- Create **two projects**:
  1. `doseedo-studio-prod` — platform "React"
  2. `stemphonic-backend-prod` — platform "Python / Flask"
- In each project → Settings → Client Keys (DSN), grab the DSN string
  (looks like `https://abc123@o456.ingest.us.sentry.io/789`)
- Settings → Organization → Auth Tokens → create a token with scopes:
  `project:read`, `project:releases`, `org:read`. This is for CI source-
  map uploads; do NOT check it into git.

## 2. Stash the secrets where the runtime expects them

### Frontend DSN — build-time env var
The React bundle needs `REACT_APP_SENTRY_DSN` at `npm run build`
time. Set it in CI (Cloud Build, GitHub Actions, or your laptop
shell) before `deploy.sh` runs:
```bash
export REACT_APP_SENTRY_DSN=https://abc123@o456.ingest.us.sentry.io/789
export REACT_APP_SENTRY_ENV=production
```
You can also drop it into `var/www/html/doseedo-react/.env.production.local`
(which should already be in `.gitignore`) for local builds. DO NOT
put it in `.env.production` — that file IS committed.

### Backend DSN — GCP Secret Manager → Modal secret
```bash
# 1. Store DSN in Secret Manager (no trailing newline — see
#    docs/LAUNCH_RUNBOOK.md for the foot-gun story).
printf %s "https://xyz@o456.ingest.us.sentry.io/890" > /tmp/sentry-dsn.clean
gcloud secrets create doseedo-sentry --data-file=/tmp/sentry-dsn.clean
rm /tmp/sentry-dsn.clean

# 2. Pull it into Modal. stemphonic_server.py reads os.environ['SENTRY_DSN'],
#    so the Modal secret key must be named SENTRY_DSN exactly.
DSN=$(gcloud secrets versions access latest --secret=doseedo-sentry)
modal secret create doseedo-sentry "SENTRY_DSN=$DSN" "SENTRY_ENV=production"

# 3. Add the secret to the Modal app. Edit modal/modal_stemphonic.py
#    and extend the secrets=[...] list on @app.cls:
#       modal.Secret.from_name("doseedo-gate"),
#       modal.Secret.from_name("doseedo-sentry"),   # <-- add this
#    Then:
cd modal && modal deploy modal_stemphonic.py
```

### CI / CLI source-map upload credentials
`deploy.sh`'s step 0b skips the source-map upload unless all three
of these are set in the shell that runs it:
```bash
export SENTRY_AUTH_TOKEN=<token from step 1>
export SENTRY_ORG=doseedo
export SENTRY_PROJECT=doseedo-studio-prod
```
Install the CLI once:
```bash
npm install -g @sentry/cli
```
The release name defaults to the short git SHA, which matches what
the frontend sends via `REACT_APP_SENTRY_RELEASE` when you set it —
otherwise Sentry will fall back to an auto-generated release. If you
want symbolicated stacks, the two release names MUST match, so in
CI do:
```bash
export SENTRY_RELEASE=$(git rev-parse --short HEAD)
export REACT_APP_SENTRY_RELEASE=$SENTRY_RELEASE
```

## 3. Verify

### Frontend
After the first deploy with DSN set, go to doseedo.com/studio and
throw a test error in the DevTools console:
```js
throw new Error('sentry test from studio')
```
Within ~1 minute it should appear in Sentry under `doseedo-studio-prod`
with a symbolicated stack (line + file from the un-minified source,
not `main.abc123.js:1:12345`). If the stack is still minified, source-
map upload didn't run (check step 0b logs in `deploy.sh` output).

### Backend
After the Modal redeploy, trigger a capture manually:
```bash
TOKEN=$(cat /tmp/smoke_token.txt)  # or mint a fresh one
curl -X POST https://arlo--doseedo-stemphonic-stemphonic-wsgi.modal.run/api/extract-midi \
  -H "Authorization: Bearer $TOKEN" \
  -F "audioFile=@/dev/null"   # intentionally bad payload — should 500
```
Or `modal app logs doseedo-stemphonic | grep sentry` to confirm
`[sentry] initialized (env=production)` at container start.

### Product events
Open DevTools → Network, drop a file, then check Sentry's Breadcrumbs
on any error capture that happens after — you should see
`category=product, message=generation.started` etc. For standalone
issue visibility, `generation.failed` and `webgpu.init_failed` also
fire as their own info-level captureMessage.

## Session Replay

Intentionally OFF. Conflicts with the "your audio stays on your
device" copy in the privacy policy and the DAW canvas is a poor
target for a replay player. Revisit in 6+ months only if we hit a
wave of unreproducible user bugs; even then, lock it to mask
`canvas`, `input`, and block the Studio page by default.

## Rotation

Rotating the DSN follows the same pattern as `internal-secret` (see
`docs/LAUNCH_RUNBOOK.md`): write to a file with `printf %s` (no
trailing newline), add a new Secret Manager version, update the
Modal secret, force-restart Modal via `modal deploy`, and rotate the
frontend env var + redeploy the frontend bundle.
