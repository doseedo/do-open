#!/usr/bin/env bash
#
# Deploy the doseedo-react frontend to production.
#
# Run from anywhere — the script resolves its own location and works
# relative to the doseedo-react/ directory it lives in. No hardcoded
# absolute paths.
#
# What this does (in order):
#   0.  npm run build (unless --skip-build)
#   1.  gsutil rsync build/ → gs://doseedo-frontend-static    (the GCS
#       backend bucket that the GCLB url-map serves at doseedo.com/* —
#       see ops/bassify-urlmap.yaml frontend-split pathMatcher)
#   2.  Force-upload index.html + asset-manifest.json with Cache-Control
#       no-store. Their byte sizes are deterministic across builds and
#       rsync's default size-compare silently skips them — without this
#       step the bucket keeps serving the old index.html that points at
#       the previous main.[hash].js bundle, which then 404s.
#   3.  gcloud run deploy doseedo-frontend --source .   (rebuilds the
#       nginx image. Needed because /home, /api/encode-latents-bulk,
#       /separate-stems and a few other paths are routed to this Cloud
#       Run service by the url-map, NOT to the bucket — see DEPLOY.md.)
#   4.  GCP CDN cache invalidation
#   5.  Cloudflare cache purge (Cloudflare sits in front of GCP)
#   6.  Verify by comparing the main.[hash].js filename in build/index.html
#       to what doseedo.com/ actually serves
#
# Required tools: gcloud (with gsutil), curl, python3
# Required secrets: /scratch/cache/secrets/cloudflare-token (chmod 600)
#
# Flags:
#   --skip-build          skip step 0 (use the existing build/ directory)
#   --skip-cloud-run      skip step 3 (use when only React/CSS changed,
#                         no nginx.conf / Dockerfile / login.html edit)
#   --skip-cloudflare     skip step 5 (use when CF token isn't on disk)
#   --dry-run             print commands instead of running them
#
# See DEPLOY.md (next to this script) for the full architecture and a
# full troubleshooting guide.

set -euo pipefail

# Resolve REPO to the directory this script lives in, regardless of cwd
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$SCRIPT_DIR"

BUCKET=gs://doseedo-frontend-static
URL_MAP=bassify
CLOUD_RUN_SERVICE=doseedo-frontend
CLOUD_RUN_REGION=us-central1
PROD_URL=https://doseedo.com
CF_ZONE=d16198032affca1d6ef0978025c5b5a0
CF_TOKEN_FILE=/scratch/cache/secrets/cloudflare-token

SKIP_BUILD=0
SKIP_CLOUD_RUN=0
SKIP_CLOUDFLARE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)      SKIP_BUILD=1 ;;
    --skip-cloud-run)  SKIP_CLOUD_RUN=1 ;;
    --skip-cloudflare) SKIP_CLOUDFLARE=1 ;;
    --dry-run)         DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
  shift
done

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

cd "$REPO"

# ── 0. build ──────────────────────────────────────────────────────────
if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "── 0. npm run build ──"
  run npm run build
else
  echo "── 0. npm run build  [SKIPPED via --skip-build] ──"
  if [[ ! -d build ]]; then
    echo "  ERROR: --skip-build set but build/ does not exist" >&2
    exit 1
  fi
fi

# ── 1. sync build/ → bucket ───────────────────────────────────────────
echo "── 1. rsync build/ → $BUCKET ──"
run gsutil -m rsync -r -c -d build/ "$BUCKET/"

# ── 2. force-upload cache-bypass HTML/manifest ────────────────────────
echo "── 2. force-upload index.html + asset-manifest.json with no-store ──"
run gsutil -h "Cache-Control:no-store, max-age=0" cp build/index.html        "$BUCKET/index.html"
run gsutil -h "Cache-Control:no-store, max-age=0" cp build/asset-manifest.json "$BUCKET/asset-manifest.json"

# ── 3. cloud run image rebuild ────────────────────────────────────────
if [[ $SKIP_CLOUD_RUN -eq 0 ]]; then
  echo "── 3. gcloud run deploy $CLOUD_RUN_SERVICE (--source .) ──"
  run gcloud run deploy "$CLOUD_RUN_SERVICE" \
    --source . \
    --region "$CLOUD_RUN_REGION" \
    --quiet
else
  echo "── 3. gcloud run deploy  [SKIPPED via --skip-cloud-run] ──"
  echo "  WARNING: nginx.conf / login.html / Dockerfile changes will NOT ship"
fi

# ── 4. invalidate GCP CDN ─────────────────────────────────────────────
echo "── 4. invalidate GCP CDN ──"
run gcloud compute url-maps invalidate-cdn-cache "$URL_MAP" --path "/*" --async

# ── 5. purge Cloudflare ───────────────────────────────────────────────
if [[ $SKIP_CLOUDFLARE -eq 0 ]]; then
  if [[ ! -r "$CF_TOKEN_FILE" ]]; then
    echo "── 5. Cloudflare purge  [SKIPPED — $CF_TOKEN_FILE missing] ──"
    echo "  WARNING: stale CSS/JS may persist at edge until next CF cache expiry"
  else
    echo "── 5. purge Cloudflare ──"
    CF_TOKEN=$(cat "$CF_TOKEN_FILE")
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] curl -X POST cloudflare purge_cache (zone $CF_ZONE)"
    else
      curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$CF_ZONE/purge_cache" \
        -H "Authorization: Bearer $CF_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"purge_everything":true}' \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('  CF purge:', d.get('success'))"
    fi
  fi
else
  echo "── 5. Cloudflare purge  [SKIPPED via --skip-cloudflare] ──"
fi

# ── 6. wait + verify bundle hash matches ──────────────────────────────
echo "── 6. wait 25s for CDN propagation, then verify ──"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] sleep 25 && curl + grep main.[hash].js comparison"
  exit 0
fi
sleep 25
EXPECTED=$(grep -oE 'main\.[a-f0-9]+\.js' build/index.html | head -1)
LIVE=$(curl -s "$PROD_URL/?cb=$RANDOM" | grep -oE 'main\.[a-f0-9]+\.js' | head -1)
echo "  expected: $EXPECTED"
echo "  live:     $LIVE"
if [[ "$EXPECTED" = "$LIVE" ]]; then
  echo "✅ deploy verified — production serving the new bundle"
  exit 0
else
  echo "⚠️  bundle mismatch — CDN may need another 30s, retry verification:"
  echo "    curl -s '$PROD_URL/?cb=\$RANDOM' | grep -oE 'main\\.[a-f0-9]+\\.js'"
  exit 3
fi
