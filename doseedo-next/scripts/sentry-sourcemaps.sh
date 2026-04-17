#!/usr/bin/env bash
# Upload production source maps to Sentry so stack traces resolve.
#
# Runs as the Next.js `postbuild` hook. No-ops in three cases:
#   - No SENTRY_AUTH_TOKEN (local dev, or the env var isn't set on Vercel)
#   - No .next build output (prebuild/sync only, no actual compile)
#   - Vercel preview builds (VERCEL_ENV != production) — we only want
#     source maps for the deploy that actually serves doseedo.com traffic
#
# Source-map upload failures ARE NOT FATAL — if Sentry's API has a hiccup
# during the upload, we don't want to take down a production deploy over
# it. The script logs the failure and exits 0 so Vercel continues.

# Don't `set -e` — we explicitly catch upload failures below.
set -uo pipefail

if [[ -z "${SENTRY_AUTH_TOKEN:-}" ]]; then
  echo "sentry-sourcemaps: SENTRY_AUTH_TOKEN not set — skipping"
  exit 0
fi

if [[ ! -d ".next" ]]; then
  echo "sentry-sourcemaps: no .next output — skipping"
  exit 0
fi

# Only upload on production deploys. Preview/dev builds would pollute the
# release history.
if [[ "${VERCEL_ENV:-}" != "production" && -n "${VERCEL_ENV:-}" ]]; then
  echo "sentry-sourcemaps: VERCEL_ENV=${VERCEL_ENV} (not production) — skipping"
  exit 0
fi

SENTRY_ORG="${SENTRY_ORG:-arloerwincom}"
SENTRY_PROJECT="${SENTRY_PROJECT:-doseedo-studio-prod}"
RELEASE="${SENTRY_RELEASE:-${VERCEL_GIT_COMMIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo dev)}}"

echo "sentry-sourcemaps: org=${SENTRY_ORG} project=${SENTRY_PROJECT} release=${RELEASE}"

# Each sentry-cli step is allowed to fail without taking down the build.
# We report status at the end so a failure is visible in Vercel logs.
upload_ok=true
run_sentry() {
  if ! npx --yes @sentry/cli "$@"; then
    upload_ok=false
    echo "sentry-sourcemaps: WARNING — sentry-cli step failed: $*"
  fi
}

run_sentry releases --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" new "$RELEASE"
run_sentry sourcemaps --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" inject .next
run_sentry sourcemaps --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" upload --release "$RELEASE" .next
run_sentry releases --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" finalize "$RELEASE"

# Strip the .map files from the served bundle so we don't leak un-minified
# source. Sentry already has them; the browser uses the `sourceMappingURL`
# comment baked in by `sentry-cli sourcemaps inject` to resolve symbols
# server-side via the release, not the client's .map download.
find .next/static -name "*.map" -type f -delete 2>/dev/null || true

if $upload_ok; then
  echo "sentry-sourcemaps: done (all steps succeeded)"
else
  echo "sentry-sourcemaps: done with warnings (build NOT failed — stack traces for this release may be minified)"
fi
exit 0
