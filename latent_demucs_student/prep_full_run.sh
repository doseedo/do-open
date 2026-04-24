#!/bin/bash
set -euo pipefail
echo "=== LATENT DEMUCS 6-STEM FULL PREP ==="
echo "$(date): Starting"

VENV=/scratch/stemphonic/venv/bin
SCRIPT_DIR=/mnt/data/system_home/arlo/do2/latent_demucs_student
LOG_DIR=/scratch/latent_demucs_student/logs
mkdir -p $LOG_DIR /scratch/mixesV7_latents /scratch/mixesV7 /scratch/moisesdb_latents /scratch/moisesdb_wavs

# ---- Step 1: Sync mixesV7 latents from GCS (already VAE-encoded) ----
echo "$(date): Step 1 — Sync mixesV7 latents from GCS"
gsutil -m rsync -r gs://ptxsessiondata/mixesV7_latents/ /scratch/mixesV7_latents/ \
    2>&1 | tail -5
echo "mixesV7_latents: $(find /scratch/mixesV7_latents -name '*.vae.pt' 2>/dev/null | wc -l) files"

# ---- Step 2: Sync mixesV7 full_mix.flac files from GCS ----
echo "$(date): Step 2 — Sync mixesV7 flac files"
find /scratch/mixesV7_latents -name 'full_mix.vae.pt' | while read f; do
    rel_dir=$(dirname "$f" | sed 's|/scratch/mixesV7_latents/||')
    local_dir="/scratch/mixesV7/$rel_dir"
    mkdir -p "$local_dir"
    if [ ! -f "$local_dir/full_mix.flac" ]; then
        gsutil -q cp "gs://ptxsessiondata/mixesV7/$rel_dir/full_mix.flac" "$local_dir/" 2>/dev/null || echo "  miss: $rel_dir"
    fi
done
echo "flacs: $(find /scratch/mixesV7 -name 'full_mix.flac' 2>/dev/null | wc -l) files"

# ---- Step 3: Build 6-stem target latents from GT stems in Latents2 ----
echo "$(date): Step 3 — Build 6-stem target latents from real GT stems"
$VENV/python3 -u $SCRIPT_DIR/build_stem6_targets.py \
    --manifest /home/arlo/gcs-bucket/DO1ckpts/master_manifest_v2.6.json \
    2>&1 | tee $LOG_DIR/build_stem6.log

# ---- Step 4: Split MUSDB 'other' stem into guitar/piano/other via htdemucs_6s ----
echo "$(date): Step 4 — Split MUSDB 'other' stems (htdemucs_6s → guitar/piano/other)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $VENV/python3 -u $SCRIPT_DIR/split_musdb_other.py \
    2>&1 | tee $LOG_DIR/split_musdb_other.log

echo ""
echo "$(date): === ALL PREP DONE ==="
echo "mixesV7 latents:  $(find /scratch/mixesV7_latents -name '*.vae.pt' 2>/dev/null | wc -l) files"
echo "mixesV7 stem6:    $(find /scratch/mixesV7_latents -name 'stem6_*.vae.pt' 2>/dev/null | wc -l) files"
echo "mixesV7 flacs:    $(find /scratch/mixesV7 -name 'full_mix.flac' 2>/dev/null | wc -l) files"
echo "MUSDB latents:    $(find /scratch/musdb18_latents -name '*.vae.pt' 2>/dev/null | wc -l) files"
echo "MUSDB 6-stem:     $(find /scratch/musdb18_latents -name 'guitar.vae.pt' 2>/dev/null | wc -l) tracks"
echo ""
echo "Ready to train. Run:"
echo "  $VENV/python3 -u $SCRIPT_DIR/train_distill_6.py --steps 30000 --batch 2 --crop_frames 200"
