#!/usr/bin/env bash
# Upload production source maps to Sentry so stack traces resolve.
#
# Runs as the Next.js `postbuild` hook. No-ops in three cases:
#   - No SENTRY_AUTH_TOKEN (local dev, or the env var isn't set on Vercel)
#   - No .next build output (prebuild/sync only, no actual compile)
#   - Vercel preview builds (VERCEL_ENV != production) — we only want
#     source maps for the deploy that actually serves doseedo.com traffic
#
# Exits 0 on no-op so it never blocks the deploy.

set -euo pipefail

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

# `sentry-cli` ships with @sentry/cli. Use npx so we don't care about PATH.
npx --yes @sentry/cli releases \
  --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" \
  new "$RELEASE"

npx --yes @sentry/cli sourcemaps \
  --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" \
  inject .next

npx --yes @sentry/cli sourcemaps \
  --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" \
  upload --release "$RELEASE" .next

npx --yes @sentry/cli releases \
  --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" \
  finalize "$RELEASE"

echo "sentry-sourcemaps: done"
