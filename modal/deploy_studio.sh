#!/bin/bash
# One-shot deploy script for doseedo studio
# Run after GCP billing is restored.
#
# Steps:
#   1. Sync React build to GCS bucket (checksum mode, never size-skip)
#   2. Force-upload index.html + asset-manifest.json (their byte sizes
#      are deterministic across builds and rsync's default size-compare
#      will silently skip them)
#   3. Cloud Run redeploy (rebuilds nginx + bakes the SPA into the image)
#   4. GCP CDN cache invalidation
#   5. Cloudflare cache purge via API token
#   6. Verify production loads the new bundle
#
# Required env on disk:
#   - /scratch/cache/secrets/cloudflare-token (chmod 600)
#
# Required tools: gcloud, gsutil, curl, python3
set -e

REPO=/scratch/do-repo/var/www/html/doseedo-react
BUCKET=gs://doseedo-frontend-static
CF_ZONE=d16198032affca1d6ef0978025c5b5a0
CF_TOKEN=$(cat /scratch/cache/secrets/cloudflare-token)

echo "── 0. sync server-side files (stemphonic + chat agent) ──"
cp /home/arlo/do2/stemphonic_server.py /scratch/stemphonic/stemphonic_server.py
cp /home/arlo/do2/chat_agent_server.py /scratch/stemphonic/chat_agent_server.py
gsutil cp /home/arlo/do2/stemphonic_server.py gs://doseedo-production/stemphonic/scripts/stemphonic_server.py
gsutil cp /home/arlo/do2/chat_agent_server.py gs://doseedo-production/stemphonic/scripts/chat_agent_server.py
echo "  (chat_agent_server is running on port 8766; restart manually if you changed it)"

echo "── 1. sync build/ → bucket (checksum) ──"
cd "$REPO"
gsutil -m rsync -r -c -d build/ "$BUCKET/"

echo "── 2. force-upload index.html + asset-manifest.json ──"
gsutil -h "Cache-Control:no-store, max-age=0" cp build/index.html "$BUCKET/index.html"
gsutil -h "Cache-Control:no-store, max-age=0" cp build/asset-manifest.json "$BUCKET/asset-manifest.json"

echo "── 3. cloud run deploy (nginx + image) ──"
gcloud run deploy doseedo-frontend --source . --region us-central1 --quiet

echo "── 4. invalidate GCP CDN ──"
gcloud compute url-maps invalidate-cdn-cache bassify --path "/*"

echo "── 5. purge Cloudflare ──"
curl -s -X POST "https://api.cloudflare.com/client/v4/zones/$CF_ZONE/purge_cache" \
  -H "Authorization: Bearer $CF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"purge_everything":true}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('CF purge:', d.get('success'))"

echo "── 6. wait + verify ──"
sleep 25
EXPECTED=$(grep -oE 'main\.[a-f0-9]+\.js' build/index.html | head -1)
LIVE=$(curl -s "https://doseedo.com/?cb=$RANDOM" | grep -oE 'main\.[a-f0-9]+\.js' | head -1)
echo "  expected: $EXPECTED"
echo "  live:     $LIVE"
if [ "$EXPECTED" = "$LIVE" ]; then
  echo "✅ deploy verified — production serving new bundle"
else
  echo "⚠️  bundle mismatch — check GCP CDN propagation (may take another 30s)"
fi
