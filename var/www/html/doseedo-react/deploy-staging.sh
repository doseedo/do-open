#!/usr/bin/env bash
#
# Deploy the frontend to STAGING. Separate from deploy.sh (prod) so
# there's no way to --env-flag-your-way into overwriting prod.
#
# What this does:
#   1. npm run build (with REACT_APP_ENV=staging so the bundle branches
#      correctly once env-aware code lands)
#   2. rsync build/ → gs://doseedo-frontend-static-staging
#   3. gcloud run deploy doseedo-frontend-staging with Dockerfile.staging
#      (nginx proxies /api/* to the staging Modal app URL)
#   4. print the Cloud Run URL so you can tell someone to go hit it
#
# No DNS and no Cloudflare purge — staging is accessed via the raw
# Cloud Run URL today. Add staging.doseedo.com Cloudflare record and
# a purge step here once DNS is provisioned.
#
# Flags:
#   --skip-build       use existing build/
#   --dry-run          print commands instead of running them

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BUCKET=gs://doseedo-frontend-static-staging
CLOUD_RUN_SERVICE=doseedo-frontend-staging
CLOUD_RUN_REGION=us-central1

SKIP_BUILD=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1 ;;
    --dry-run)    DRY_RUN=1 ;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

run() { if [[ $DRY_RUN -eq 1 ]]; then echo "[dry-run] $*"; else "$@"; fi; }

# ── 0. build ──────────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "── 0. npm run build (staging) ──"
  REACT_APP_ENV=staging run npm run build
else
  echo "── 0. npm run build  [SKIPPED] ──"
fi

# Staging bucket rule matches prod: NO .map files in public storage
# even if sentry upload is off. Strip before rsync.
run rm -f build/static/js/*.map

# ── 1. sync build/ → staging bucket ───────────────────────────────────
echo "── 1. rsync build/ → $BUCKET ──"
run gsutil -m rsync -r -c -d build/ "$BUCKET/"

# ── 2. force-upload cache-bypass HTML/manifest ────────────────────────
echo "── 2. force-upload index.html + asset-manifest.json with no-store ──"
run gsutil -h "Cache-Control:no-store, max-age=0" cp build/index.html        "$BUCKET/index.html"
run gsutil -h "Cache-Control:no-store, max-age=0" cp build/asset-manifest.json "$BUCKET/asset-manifest.json"

# ── 3. cloud run deploy with staging Dockerfile ──────────────────────
# gcloud run --source auto-detects ./Dockerfile. We can't pass a
# different path, so temporarily swap Dockerfile.staging into place,
# deploy, then restore. Trap on exit so a crash doesn't leave prod
# Dockerfile shadowed.
echo "── 3. gcloud run deploy $CLOUD_RUN_SERVICE (staging Dockerfile) ──"
mv Dockerfile Dockerfile.prod.tmp
cp Dockerfile.staging Dockerfile
trap 'mv Dockerfile.prod.tmp Dockerfile 2>/dev/null || true' EXIT

run gcloud run deploy "$CLOUD_RUN_SERVICE" \
  --source . \
  --region "$CLOUD_RUN_REGION" \
  --allow-unauthenticated \
  --set-env-vars "DOSEEDO_ENV=staging" \
  --quiet

# Restore prod Dockerfile immediately — don't rely on trap alone.
rm -f Dockerfile
mv Dockerfile.prod.tmp Dockerfile
trap - EXIT

echo "── 4. fetching staging URL ──"
URL=$(gcloud run services describe "$CLOUD_RUN_SERVICE" --region="$CLOUD_RUN_REGION" --format="value(status.url)" 2>/dev/null || true)
echo "✅ staging deployed: $URL"
echo "   bundle:    $(grep -oE 'main\.[a-f0-9]+\.js' build/index.html | head -1)"
echo "   test with: curl -I $URL | grep -i x-doseedo-env"
