#!/usr/bin/env python3
"""
Generate violin audio samples using TimbreVAE

Finds violin samples, encodes to timbre space, then decodes back to audio.
Also generates novel timbres by interpolating/sampling.
"""

import sys
sys.path.insert(0, '/home/arlo/Data/dø')
sys.path.append('/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/soundspace')

import torch
import torchaudio
from pathlib import Path
import json
import orjson
import random

from timbre_vae import TimbreVAE

print("=" * 60)
print("Violin Audio Generation with TimbreVAE")
print("=" * 60)

OUTPUT_DIR = Path("/home/arlo/soundspace_data/violin_test")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============== Load TimbreVAE ==============
print("\n[1] Loading TimbreVAE...")
VAE_PATH = "/home/arlo/soundspace_checkpoints/timbre_vae_final.pt"
vae_ckpt = torch.load(VAE_PATH, map_location='cpu', weights_only=False)
vae_config = vae_ckpt['config']

vae = TimbreVAE(**{k: v for k, v in vae_config.items()
                   if k in ['input_dim', 'latent_dim', 'hidden_dim',
                           'n_residual', 'dropout', 'student_t_df']})
vae.load_state_dict(vae_ckpt['model'])
vae.eval()
vae_mean = vae_ckpt['mean']
vae_std = vae_ckpt['std']
print(f"✓ TimbreVAE loaded: {vae_config['latent_dim']}D")

# ============== Load DCAE for decoding ==============
print("\n[2] Loading DCAE...")

CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints"
from acestep.pipeline_ace_step import ACEStepPipeline
pipeline = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR)
pipeline.load_checkpoint(CHECKPOINT_DIR)
dcae = pipeline.music_dcae.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dcae = dcae.to(DEVICE)
print(f"✓ DCAE loaded on {DEVICE}")

# ============== Load manifest to find actual latent files ==============
print("\n[3] Loading manifest to find latent files...")
MANIFEST_PATH = "/home/arlo/gcs-bucket/Manifests/unified_manifest.json"
LOCAL_MANIFEST = "/tmp/unified_manifest.json"

# Copy from GCS to local for faster loading
import subprocess
import os
if not os.path.exists(LOCAL_MANIFEST) or os.path.getmtime(LOCAL_MANIFEST) < os.path.getmtime(MANIFEST_PATH):
    print("  Copying manifest to local...")
    subprocess.run(["cp", MANIFEST_PATH, LOCAL_MANIFEST], check=True)

print("  Loading with orjson...")
with open(LOCAL_MANIFEST, 'rb') as f:
    manifest = orjson.loads(f.read())

entries = manifest.get('entries', [])
# Don't check exists() here - it's slow on GCS. Check when loading instead.
entries_with_latent = [
    e for e in entries
    if e.get('has_latent') == True and e.get('latent_path')
]
print(f"✓ Found {len(entries_with_latent)} entries with latent paths")

# ============== Helper to load latent file ==============
def load_latent_file(latent_path: str) -> torch.Tensor:
    """Load a single latent file and return frames [T, 128]."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            for key in ['latents', 'latent', 'z', 'embedding']:
                if key in data and isinstance(data[key], torch.Tensor):
                    data = data[key]
                    break
        if not isinstance(data, torch.Tensor):
            return None
        if data.dim() == 4:
            data = data.squeeze(0)
        if data.dim() != 3:
            return None
        C, H, T = data.shape
        if C != 8 or H != 16:
            return None
        # Flatten to frames: [T, 128]
        frames = data.permute(2, 0, 1).reshape(T, C * H)
        return frames.float()
    except Exception as e:
        return None

# ============== Sample some latent files for analysis ==============
print("\n[4] Encoding frames from random files to analyze timbre space...")
random.seed(42)
sample_entries = random.sample(entries_with_latent, min(100, len(entries_with_latent)))

all_frames = []
for entry in sample_entries:
    frames = load_latent_file(entry['latent_path'])
    if frames is not None and len(frames) >= 32:
        all_frames.append(frames[:100])  # Take first 100 frames from each

sample_frames = torch.cat(all_frames, dim=0)
print(f"  Loaded {len(sample_frames)} frames from {len(all_frames)} files")

# Normalize and encode
frames_norm = (sample_frames - vae_mean) / (vae_std + 1e-8)
with torch.no_grad():
    mu, logvar = vae.encode(frames_norm)

print(f"  Encoded {len(sample_frames)} frames")
print(f"  Timbre space range: [{mu.min():.2f}, {mu.max():.2f}]")

# ============== Generate audio from different timbre regions ==============
print("\n[5] Generating audio samples...")

SAMPLE_RATE = 48000

def frames_to_audio(frames_128d, dcae, vae_mean, vae_std, sr=SAMPLE_RATE):
    """Convert 128D frames back to audio via DCAE."""
    # frames_128d: [N, 128]
    T = frames_128d.shape[0]

    # Reshape to DCAE format [1, 8, 16, T]
    latents = frames_128d.view(T, 8, 16).permute(1, 2, 0).unsqueeze(0)  # [1, 8, 16, T]
    latents = latents.to(DEVICE)

    # Estimate audio length
    estimated_len = int(T * 4096 * sr / 44100)
    audio_len = torch.tensor([estimated_len], device=DEVICE)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, wavs = dcae.decode_overlap(latents, audio_lengths=audio_len, sr=sr)

    return wavs[0].cpu()  # [samples]

# ============== Pre-compute z values for many recordings ==============
print("  Pre-computing z values for recordings...")
# Load more files to have good variety for matching
n_files_to_index = 500
random.seed(123)  # Different seed for variety
index_entries = random.sample(entries_with_latent, min(n_files_to_index, len(entries_with_latent)))

file_z_list = []  # List of (entry, mean_z, frames)
for entry in index_entries:
    frames = load_latent_file(entry['latent_path'])
    if frames is None or len(frames) < 256:
        continue
    # Take middle chunk
    start = len(frames) // 4
    chunk = frames[start:start + 256]
    chunk_energy = chunk.pow(2).sum(dim=-1).mean()
    if chunk_energy < 10.0:  # Skip quiet recordings
        continue
    # Compute mean z for this recording
    chunk_norm = (chunk - vae_mean) / (vae_std + 1e-8)
    with torch.no_grad():
        mu_chunk, _ = vae.encode(chunk_norm)
    mean_z = mu_chunk.mean(dim=0)  # [8]
    file_z_list.append((entry, mean_z, chunk))

print(f"  Indexed {len(file_z_list)} recordings with z values")

def find_nearest_recording(target_z, file_z_list, exclude_indices=None):
    """Find recording with z closest to target_z."""
    exclude_indices = exclude_indices or set()
    best_dist = float('inf')
    best_idx = 0
    for i, (entry, mean_z, chunk) in enumerate(file_z_list):
        if i in exclude_indices:
            continue
        dist = (mean_z - target_z).pow(2).sum().item()
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx, file_z_list[best_idx]

# 1. Generate from cluster centers - find recordings closest to each center
print("  Generating from cluster centers (nearest real recordings)...")
from sklearn.cluster import KMeans

all_z = torch.stack([z for _, z, _ in file_z_list])
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(all_z.numpy())
centers = torch.from_numpy(kmeans.cluster_centers_)

used_indices = set()
for i, center in enumerate(centers):
    idx, (entry, mean_z, chunk) = find_nearest_recording(center, file_z_list, used_indices)
    used_indices.add(idx)
    audio = frames_to_audio(chunk, dcae, vae_mean, vae_std)
    out_path = OUTPUT_DIR / f"cluster_{i:02d}.wav"
    torchaudio.save(str(out_path), audio.unsqueeze(0) if audio.dim() == 1 else audio, 44100)
    group = entry.get('group', '?')
    print(f"    ✓ Saved {out_path.name} ({group})")

# 2. Generate from prior - find recordings with extreme/unusual z values
print("  Generating from prior samples (diverse real recordings)...")
# Sample diverse recordings by taking from different regions
z_norms = torch.stack([z for _, z, _ in file_z_list]).norm(dim=1)
sorted_indices = z_norms.argsort(descending=True)
for i in range(5):
    idx = sorted_indices[i * len(sorted_indices) // 5].item()  # Spread across distribution
    entry, mean_z, chunk = file_z_list[idx]
    audio = frames_to_audio(chunk, dcae, vae_mean, vae_std)
    out_path = OUTPUT_DIR / f"prior_{i:02d}.wav"
    torchaudio.save(str(out_path), audio.unsqueeze(0) if audio.dim() == 1 else audio, 44100)
    group = entry.get('group', '?')
    print(f"    ✓ Saved {out_path.name} ({group})")

# 3. Interpolation - show 5 recordings along path between two clusters
print("  Generating interpolation (recordings along cluster path)...")
z1, z2 = centers[0], centers[4]  # Most different clusters
for i, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    target_z = (1 - alpha) * z1 + alpha * z2
    idx, (entry, mean_z, chunk) = find_nearest_recording(target_z, file_z_list)
    audio = frames_to_audio(chunk, dcae, vae_mean, vae_std)
    out_path = OUTPUT_DIR / f"interp_{i:02d}_a{int(alpha*100):03d}.wav"
    torchaudio.save(str(out_path), audio.unsqueeze(0) if audio.dim() == 1 else audio, 44100)
    group = entry.get('group', '?')
    print(f"    ✓ Saved {out_path.name} ({group})")

# 4. Roundtrip: encode real frames, decode back
# IMPORTANT: Load ACTUAL latent files and use consecutive frames from SAME recording
print("  Generating roundtrips from real latent files (consecutive frames)...")

chunk_size = 256
n_roundtrips = 5
roundtrip_count = 0

# Shuffle entries to get variety
random.shuffle(entries_with_latent)

for entry in entries_with_latent:
    if roundtrip_count >= n_roundtrips:
        break

    frames = load_latent_file(entry['latent_path'])
    if frames is None or len(frames) < chunk_size:
        continue

    # Take a chunk from middle of the file (avoid intro/outro silence)
    start = len(frames) // 4
    if start + chunk_size > len(frames):
        start = 0
    chunk = frames[start:start + chunk_size]  # [256, 128] - CONSECUTIVE frames from SAME file

    # Check energy
    chunk_energy = chunk.pow(2).sum(dim=-1).mean()
    if chunk_energy < 5.0:
        continue

    chunk_norm = (chunk - vae_mean) / (vae_std + 1e-8)

    # Encode through VAE - use mu directly (no reparameterize noise) for cleaner reconstruction
    with torch.no_grad():
        mu_chunk, _ = vae.encode(chunk_norm)
        recon_norm = vae.decode(mu_chunk)  # Use mu directly, not reparameterize

    recon = recon_norm * (vae_std + 1e-8) + vae_mean

    # Original audio (decode original latents)
    audio_orig = frames_to_audio(chunk, dcae, vae_mean, vae_std)
    out_path = OUTPUT_DIR / f"roundtrip_{roundtrip_count:02d}_original.wav"
    torchaudio.save(str(out_path), audio_orig.unsqueeze(0) if audio_orig.dim() == 1 else audio_orig, 44100)

    # Reconstructed audio (decode VAE-reconstructed latents)
    audio_recon = frames_to_audio(recon, dcae, vae_mean, vae_std)
    out_path = OUTPUT_DIR / f"roundtrip_{roundtrip_count:02d}_reconstructed.wav"
    torchaudio.save(str(out_path), audio_recon.unsqueeze(0) if audio_recon.dim() == 1 else audio_recon, 44100)

    group = entry.get('group', 'unknown')
    subgroup = entry.get('subgroup', 'unknown')
    print(f"    ✓ Saved roundtrip_{roundtrip_count:02d}_*.wav ({group}/{subgroup}, energy={chunk_energy:.1f})")
    roundtrip_count += 1

print(f"\n✅ All samples saved to {OUTPUT_DIR}")
print(f"\nListen at: https://doseedo.com/space/ (Listening Test section)")
print(f"Or directly: ls {OUTPUT_DIR}/*.wav")
