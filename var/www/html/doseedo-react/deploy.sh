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
    --skip-git)        SKIP_GIT=1 ;;
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

# ── pre-flight: git pull + dirty-tree guard ───────────────────────────
# Root cause of the 2026-04-12 deploy collision: someone synced a stale
# local build/ to the bucket because they hadn't pulled the latest
# commits. Two guards:
#
#   1. Pull from origin so the build always reflects the latest pushed
#      source. Fails-fast if there's a merge conflict — fix locally
#      before retrying.
#
#   2. Warn (but don't block) if there are uncommitted changes. Those
#      changes won't be in git and the NEXT person who deploys without
#      them will overwrite the bucket. The right fix is to commit first,
#      but we don't hard-block because a quick CSS iteration might
#      legitimately run deploy before committing. --skip-git bypasses
#      both checks.
if [[ "${SKIP_GIT:-0}" -eq 0 ]]; then
  echo "── pre-flight: git pull + dirty-tree check ──"
  if ! run git pull --ff-only origin main 2>&1; then
    echo "  ERROR: git pull failed (merge conflict or non-fast-forward)." >&2
    echo "  Resolve locally, commit, then retry deploy." >&2
    exit 1
  fi
  DIRTY=$(git status --porcelain --untracked-files=no 2>/dev/null | wc -l)
  if [[ "$DIRTY" -gt 0 ]]; then
    echo "  ⚠  WARNING: $DIRTY uncommitted change(s) in the working tree."
    echo "  These changes WILL be built and deployed, but won't survive"
    echo "  the next deploy by someone else (since they're not in git)."
    echo "  Consider committing before deploying."
    echo "  (set SKIP_GIT=1 to suppress this check)"
  fi
else
  echo "── pre-flight: git pull + dirty-tree check  [SKIPPED via SKIP_GIT=1] ──"
fi

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

# ── 0b. upload source maps to Sentry + strip them from build/ ────────
# Source maps are ESSENTIAL for debugging prod errors — Sentry line
# numbers through the minified bundle are otherwise useless. But we
# DON'T want to publish .map files alongside the JS in the GCS bucket:
# that would leak un-minified source and defeats the point of minifying.
#
# Steps:
#   1. sentry-cli releases new $SENTRY_RELEASE
#   2. sentry-cli sourcemaps upload build/static/js  (inject + upload)
#   3. rm build/static/js/*.map so the next rsync doesn't push them
#
# NO-OPs if SENTRY_AUTH_TOKEN + SENTRY_ORG + SENTRY_PROJECT aren't set.
# That lets you run deploy.sh locally without Sentry creds while still
# failing loudly in the CI / prod-deploy path (set those three and the
# upload runs automatically).
if [[ -n "${SENTRY_AUTH_TOKEN:-}" && -n "${SENTRY_ORG:-}" && -n "${SENTRY_PROJECT:-}" ]]; then
  echo "── 0b. sentry source-map upload ──"
  if ! command -v sentry-cli >/dev/null 2>&1; then
    echo "  ERROR: sentry-cli not installed. Install via: npm i -g @sentry/cli" >&2
    exit 1
  fi
  SENTRY_RELEASE="${SENTRY_RELEASE:-$(git rev-parse --short HEAD 2>/dev/null || date +%s)}"
  echo "  release=$SENTRY_RELEASE"
  run sentry-cli releases --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" new "$SENTRY_RELEASE"
  run sentry-cli sourcemaps --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" \
      inject --release "$SENTRY_RELEASE" build/static/js
  run sentry-cli sourcemaps --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" \
      upload --release "$SENTRY_RELEASE" build/static/js
  run sentry-cli releases --org "$SENTRY_ORG" --project "$SENTRY_PROJECT" finalize "$SENTRY_RELEASE"
  # Strip map files so they DON'T get rsynced to the public bucket.
  run rm -f build/static/js/*.map
else
  echo "── 0b. sentry source-map upload  [SKIPPED — set SENTRY_AUTH_TOKEN + SENTRY_ORG + SENTRY_PROJECT] ──"
  # Still strip .map files so we never accidentally publish them even
  # without a Sentry setup — privacy > debuggability.
  run rm -f build/static/js/*.map
fi

# ── 1. sync build/ → bucket ───────────────────────────────────────────
# Sync each subdirectory explicitly using `gcloud storage rsync` (avoids
# the gsutil credential issues on macOS and the pattern-matching bug that
# caused gcloud storage rsync to skip static/js/ chunks when a top-level
# rsync was used with an exclude regex — 2026-04-16 incident).
#
# Directories that live in the bucket but NOT in build/ (must not be wiped):
#   static/models/  — ONNX model files served from GCS
#   models/         — legacy path, also GCS-only
#   assets/icons/   — instrument icon PNGs, GCS-only
#   static/media/   — WASM runtime (24 MB), GCS-only
echo "── 1. rsync build/static/js → $BUCKET/static/js ──"
run gcloud storage rsync --recursive --delete-unmatched-destination-objects \
  build/static/js/ "$BUCKET/static/js/"

echo "── 1b. rsync build/static/css → $BUCKET/static/css ──"
run gcloud storage rsync --recursive --delete-unmatched-destination-objects \
  build/static/css/ "$BUCKET/static/css/"

echo "── 1c. rsync build root (html, manifest, favicon, workers…) → $BUCKET ──"
# Top-level rsync with no --delete so it never touches static/ or models/
run gcloud storage rsync --recursive \
  --exclude="^static/|^models/|^assets/" \
  build/ "$BUCKET/"

# ── 2. force-upload cache-bypass HTML/manifest ────────────────────────
echo "── 2. force-upload index.html + asset-manifest.json with no-store ──"
run gcloud storage cp --cache-control="no-store, max-age=0" build/index.html        "$BUCKET/index.html"
run gcloud storage cp --cache-control="no-store, max-age=0" build/asset-manifest.json "$BUCKET/asset-manifest.json"

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
