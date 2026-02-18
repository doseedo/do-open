#!/usr/bin/env python3
"""
Generate training pairs for the Audio Domain Editor.

For each sample:
  1. Load z from raw latent file, crop to 32 frames (~3s)
  2. Decode z → original audio via DCAE
  3. For each axis × delta: z_edited = z + delta * scale * axis → decode → edited audio
  4. Save (original_audio, edited_audio, axis_idx, delta) pairs

Training an AudioDomainEditor on these pairs lets us apply axis edits
directly in audio/STFT domain, bypassing DCAE decode entirely at inference.
No frame boundary artifacts, original fidelity preserved.
"""

import sys
import torch
import numpy as np
import orjson
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
PITCHBIN_AXES_PATH = SCRIPT_DIR.parent / "test_outputs" / "pitchbin_discovery" / "pitchbin_axes.pt"
DATA_CACHE_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "data_cache.pt"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "audio_domain_editor"

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SR = 44100
Z_DIM = 128
Z_FRAMES = 32         # ~3 seconds per clip
N_SAMPLES = 150        # number of distinct samples
TOP_AXES = 6           # top 6 pitchbin axes
DELTAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
DECODE_BATCH = 6       # DCAE decode batch size


def load_dcae(device):
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_CKPT,
        vocoder_checkpoint_path=VOCODER_CKPT,
    ).to(device).eval()
    return dcae


def load_axes():
    print("Loading pitchbin axes...")
    data = torch.load(PITCHBIN_AXES_PATH, weights_only=False, map_location='cpu')
    axes = []
    for ax in data['pitchbin_pca'][:TOP_AXES]:
        direction = ax['direction']
        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction).float()
        direction = direction / (direction.norm() + 1e-10)
        axes.append(direction)
    return torch.stack(axes)  # [N_axes, 128]


def compute_safe_scales(axes, z_samples):
    """Compute per-axis std of projections = 1 slider unit."""
    # z_samples: [N, 128]
    projections = z_samples @ axes.T  # [N, N_axes]
    scales = projections.std(dim=0)   # [N_axes]
    print(f"  Safe scales: {[f'{s:.3f}' for s in scales.tolist()]}")
    return scales


def get_latent_paths_from_cache():
    """Get latent paths from data cache via manifest mapping."""
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    sms_to_latent = {}
    for entry in manifest['entries']:
        sms_to_latent[entry['path']] = entry['latent_path']

    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')

    paths = []
    seen = set()
    for sample in cache:
        lp = sms_to_latent.get(sample['path'])
        if lp and lp not in seen:
            seen.add(lp)
            paths.append(lp)
    return paths


def load_z_crop(path, n_frames=Z_FRAMES):
    """Load latent file → z_flat [n_frames, 128]."""
    try:
        d = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(d, dict) and 'latents' in d:
            z = d['latents']
        elif isinstance(d, torch.Tensor):
            z = d
        else:
            return None
        if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16:
            return None
        T = z.shape[2]
        if T < n_frames:
            return None
        # Take middle crop
        start = (T - n_frames) // 2
        z_crop = z[:, :, start:start + n_frames]  # [8, 16, n_frames]
        return z_crop.float()
    except Exception:
        return None


def decode_batch(dcae, z_batch_4d, device):
    """Decode a batch of z [B, 8, 16, T] → list of audio numpy arrays."""
    B, _, _, T = z_batch_4d.shape
    audio_len = int(T * SR / 10.8)
    audio_lengths = torch.tensor([audio_len] * B, device=device)

    with torch.no_grad():
        sr, wavs = dcae.decode(z_batch_4d.to(device), audio_lengths=audio_lengths, sr=SR)

    results = []
    for i in range(B):
        audio = wavs[i].mean(dim=0).cpu()  # mono, float tensor
        results.append(audio)
    return results


def main():
    print("=" * 60)
    print("GENERATING AXIS EDIT TRAINING PAIRS")
    print("=" * 60)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dcae = load_dcae(device)
    axes = load_axes()  # [N_axes, 128]
    n_axes = len(axes)

    print(f"\n  {n_axes} axes, {len(DELTAS)} deltas = {n_axes * len(DELTAS)} edits per sample")

    # Get latent paths
    print("\nGetting latent paths...")
    latent_paths = get_latent_paths_from_cache()
    print(f"  {len(latent_paths)} unique latent paths from cache")

    # Load z crops
    print(f"\nLoading z crops ({Z_FRAMES} frames each)...")
    z_crops = []  # [8, 16, T] each
    valid_paths = []

    for lp in latent_paths:
        if len(z_crops) >= N_SAMPLES:
            break
        z = load_z_crop(lp)
        if z is not None:
            z_crops.append(z)
            valid_paths.append(lp)
            if len(z_crops) % 25 == 0:
                print(f"  Loaded {len(z_crops)}/{N_SAMPLES}")

    print(f"  Loaded {len(z_crops)} z crops")

    # Compute safe scales from all z values
    z_flat_all = torch.stack([
        z.permute(2, 0, 1).reshape(Z_FRAMES, Z_DIM) for z in z_crops
    ]).reshape(-1, Z_DIM)  # [N*T, 128]
    safe_scales = compute_safe_scales(axes, z_flat_all)

    # Pre-scale axes: axis_scaled[i] = safe_scale[i] * axis_direction[i]
    axes_scaled = axes * safe_scales.unsqueeze(1)  # [N_axes, 128]

    # ============================================================
    # Decode originals
    # ============================================================
    print(f"\nDecoding {len(z_crops)} originals...")
    original_audios = []

    for batch_start in range(0, len(z_crops), DECODE_BATCH):
        batch_end = min(batch_start + DECODE_BATCH, len(z_crops))
        z_batch = torch.stack(z_crops[batch_start:batch_end])  # [B, 8, 16, T]
        audios = decode_batch(dcae, z_batch, device)
        original_audios.extend(audios)
        if (batch_start + DECODE_BATCH) % 25 < DECODE_BATCH:
            print(f"  Decoded {min(batch_end, len(z_crops))}/{len(z_crops)} originals")

    # Trim all to same length
    min_len = min(a.shape[0] for a in original_audios)
    original_audios = [a[:min_len] for a in original_audios]
    print(f"  Audio length: {min_len} samples ({min_len/SR:.2f}s)")

    # ============================================================
    # Decode edits
    # ============================================================
    n_edits_per_sample = n_axes * len(DELTAS)
    total_edits = len(z_crops) * n_edits_per_sample
    print(f"\nDecoding {total_edits} edits ({n_edits_per_sample} per sample)...")

    edited_audios = []    # flat list of audio tensors
    edit_sample_idx = []  # which original sample
    edit_axis_idx = []    # which axis
    edit_delta = []       # delta value

    edits_done = 0

    for sample_i, z_4d in enumerate(z_crops):
        # z_4d: [8, 16, T]
        z_flat = z_4d.permute(2, 0, 1).reshape(1, Z_FRAMES, Z_DIM)  # [1, T, 128]

        # Build all edits for this sample
        edit_z_list = []
        edit_info = []

        for axis_i in range(n_axes):
            for delta in DELTAS:
                z_edited = z_flat + delta * axes_scaled[axis_i].unsqueeze(0).unsqueeze(0)
                # Back to [8, 16, T]
                z_edited_4d = z_edited.reshape(Z_FRAMES, 8, 16).permute(1, 2, 0)
                edit_z_list.append(z_edited_4d)
                edit_info.append((sample_i, axis_i, delta))

        # Decode in batches
        for batch_start in range(0, len(edit_z_list), DECODE_BATCH):
            batch_end = min(batch_start + DECODE_BATCH, len(edit_z_list))
            z_batch = torch.stack(edit_z_list[batch_start:batch_end])
            audios = decode_batch(dcae, z_batch, device)

            for j, audio in enumerate(audios):
                idx = batch_start + j
                si, ai, d = edit_info[idx]
                edited_audios.append(audio[:min_len])
                edit_sample_idx.append(si)
                edit_axis_idx.append(ai)
                edit_delta.append(d)

        edits_done += len(edit_z_list)
        if (sample_i + 1) % 10 == 0:
            print(f"  Sample {sample_i + 1}/{len(z_crops)}: "
                  f"{edits_done}/{total_edits} edits decoded")

    # ============================================================
    # Save
    # ============================================================
    print(f"\nSaving {len(original_audios)} originals + {len(edited_audios)} edits...")

    save_data = {
        'original_audio': torch.stack(original_audios),     # [N_samples, L]
        'edited_audio': torch.stack(edited_audios),          # [N_edits, L]
        'edit_sample_idx': torch.tensor(edit_sample_idx, dtype=torch.long),
        'edit_axis_idx': torch.tensor(edit_axis_idx, dtype=torch.long),
        'edit_delta': torch.tensor(edit_delta, dtype=torch.float32),
        'axis_directions': axes,                             # [N_axes, 128]
        'axis_safe_scales': safe_scales,                     # [N_axes]
        'sample_rate': SR,
        'n_samples': len(original_audios),
        'n_axes': n_axes,
        'deltas': DELTAS,
        'audio_length': min_len,
    }

    save_path = OUTPUT_DIR / "training_pairs.pt"
    torch.save(save_data, save_path)

    file_size = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {save_path} ({file_size:.0f} MB)")

    print(f"\n  Summary:")
    print(f"    {len(original_audios)} samples × {n_edits_per_sample} edits = {len(edited_audios)} pairs")
    print(f"    Audio: {min_len} samples ({min_len/SR:.2f}s) at {SR} Hz")
    print(f"    Axes: {n_axes}, deltas: {DELTAS}")


if __name__ == '__main__':
    main()
