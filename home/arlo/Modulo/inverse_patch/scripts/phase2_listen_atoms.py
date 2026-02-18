#!/usr/bin/env python3
"""
Phase 2: Listen to Discovered Atoms

For each top atom in the learned dictionary, generate audio to hear
what it captures — the thing SMS can't represent.

For each test sample × top atom:
  1. z_real decoded                    → ground truth
  2. z_sms_approx decoded             → SMS-only (what's missing = residual)
  3. z_sms + this_atom_only decoded   → SMS + just this one atom
  4. z_sms + all_atoms decoded         → full reconstruction

Also: "atom solo" — add each atom at varying strengths to a neutral z
to hear the atom's effect in isolation.
"""

import sys
import torch
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
import torchaudio
import gc

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM
from phase2_residual_dictionary import SparseDictionary

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
DICT_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_residual_dictionary" / "residual_dictionary.pt"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "phase2_atom_listening"

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
HOP_SIZE = 512
Z_FLAT_DIM = 128
MAX_FRAMES = 256

N_TOP_ATOMS = 10      # listen to top 10 atoms
N_SAMPLES = 5         # test on 5 diverse samples


def decode_z_flat_to_audio(dcae, z_flat, device):
    """z_flat [1, T, 128] → audio numpy array."""
    B, T, D = z_flat.shape
    z_4d = z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)  # [1, 8, 16, T]
    audio_len = T * HOP_SIZE
    audio_lengths = torch.tensor([audio_len], device=device)
    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return wavs[0].mean(dim=0).cpu().numpy()


def save_wav(audio, path):
    """Save numpy audio as wav."""
    t = torch.from_numpy(audio).float().unsqueeze(0)
    # Normalize to prevent clipping
    peak = t.abs().max()
    if peak > 0.95:
        t = t * (0.95 / peak)
    torchaudio.save(str(path), t, SAMPLE_RATE)


def load_models(device):
    """Load Phase 1 codec, Phase 2 dictionary, and DCAE."""
    # Phase 1
    print("Loading Phase 1 codec...")
    codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_FLAT_DIM, g_hidden=384, f_hidden=256)
    codec.load_state_dict(torch.load(PHASE1_PATH, weights_only=True, map_location='cpu'))
    codec = codec.to(device).eval()

    # Phase 2 dictionary
    print("Loading Phase 2 dictionary...")
    ckpt = torch.load(DICT_PATH, weights_only=False, map_location='cpu')
    n_atoms = ckpt['n_atoms']
    top_k = ckpt.get('top_k', 8)
    dict_model = SparseDictionary(z_dim=Z_FLAT_DIM, n_atoms=n_atoms, top_k=top_k)
    dict_model.load_state_dict(ckpt['model'])
    dict_model = dict_model.to(device).eval()
    res_mean = ckpt['res_mean'].to(device)
    res_std = ckpt['res_std'].to(device)

    # DCAE
    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_CKPT,
        vocoder_checkpoint_path=VOCODER_CKPT,
    ).to(device).eval()

    return codec, dict_model, dcae, res_mean, res_std


def gather_diverse_samples(n=20):
    """Load diverse real audio latents — pick from different directories for variety."""
    print(f"\nGathering {n} diverse samples...")

    # Get latent files from different subdirectories for diversity
    subdirs = sorted(LATENT_BASE.iterdir())
    samples = []
    seen_dirs = set()

    for subdir in subdirs:
        if not subdir.is_dir():
            continue
        pt_files = list(subdir.rglob("*.pt"))
        for pt in pt_files:
            parent_name = pt.parent.name
            if parent_name in seen_dirs:
                continue

            try:
                loaded = torch.load(pt, weights_only=False, map_location='cpu')
                if not isinstance(loaded, dict) or 'latents' not in loaded:
                    continue
                z = loaded['latents']
                if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16 or z.shape[2] < 20:
                    continue
                if z.shape[2] > MAX_FRAMES:
                    z = z[:, :, :MAX_FRAMES]

                samples.append({
                    'z_real': z,
                    'path': str(pt),
                    'name': pt.stem,
                })
                seen_dirs.add(parent_name)

                if len(samples) >= n:
                    break
            except Exception:
                continue

        if len(samples) >= n:
            break

    print(f"  Loaded {len(samples)} diverse samples")
    return samples


def compute_atom_activations(codec, dict_model, sample, res_mean, res_std, device):
    """
    For one sample, compute:
    - z_flat, z_sms_approx, residual, activations, per-atom contributions
    """
    z_real = sample['z_real'].unsqueeze(0).to(device)
    z_flat = codec.z_to_flat(z_real)  # [1, T, 128]

    with torch.no_grad():
        sms_params = codec.forward_F(z_flat)
        z_sms = codec.forward_G(sms_params)
        residual = z_flat - z_sms

        residual_norm = (residual - res_mean) / res_std
        recon_norm, activations = dict_model(residual_norm)
        residual_recon = recon_norm * res_std + res_mean

    return {
        'z_flat': z_flat,
        'z_sms': z_sms,
        'residual': residual,
        'residual_recon': residual_recon,
        'activations': activations,  # [1, T, K]
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("PHASE 2: ATOM LISTENING TEST")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    codec, dict_model, dcae, res_mean, res_std = load_models(device)
    atoms = dict_model.atoms.data  # [K, 128]

    # Find top atoms by computing activations across many samples
    print("\nProfiling atom usage across samples...")
    all_samples = gather_diverse_samples(n=50)
    atom_total_activation = torch.zeros(dict_model.n_atoms, device=device)

    for sample in all_samples:
        info = compute_atom_activations(codec, dict_model, sample, res_mean, res_std, device)
        # Sum activations across all frames for this sample
        atom_total_activation += info['activations'].squeeze(0).sum(dim=0)  # [K]

    # Rank atoms
    atom_ranking = torch.argsort(atom_total_activation, descending=True)
    top_atom_ids = atom_ranking[:N_TOP_ATOMS].cpu().tolist()
    print(f"\nTop {N_TOP_ATOMS} atoms by total activation: {top_atom_ids}")
    for rank, aid in enumerate(top_atom_ids):
        print(f"  Rank {rank}: atom {aid}, total activation={atom_total_activation[aid]:.1f}")

    # Pick N_SAMPLES diverse test samples
    test_samples = all_samples[:N_SAMPLES]

    # ============================================================
    # Test 1: Per-sample decomposition
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Sample Decomposition")
    print("  For each sample: original, SMS-only, SMS+each_atom, SMS+all_atoms")
    print("=" * 60)

    for si, sample in enumerate(test_samples):
        sample_name = sample['name'][:30]
        sample_dir = OUTPUT_DIR / f"sample_{si}_{sample_name}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        info = compute_atom_activations(codec, dict_model, sample, res_mean, res_std, device)

        print(f"\n  Sample {si}: {sample_name}")

        # 1. Ground truth (z_real)
        audio_gt = decode_z_flat_to_audio(dcae, info['z_flat'], device)
        save_wav(audio_gt, sample_dir / "00_ground_truth.wav")

        # 2. SMS only (z_sms)
        audio_sms = decode_z_flat_to_audio(dcae, info['z_sms'], device)
        save_wav(audio_sms, sample_dir / "01_sms_only.wav")

        cos_sms = TF.cosine_similarity(info['z_flat'], info['z_sms'], dim=-1).mean().item()
        print(f"    SMS-only cos_sim: {cos_sms:.4f}")

        # 3. SMS + all atoms (full reconstruction)
        z_full = info['z_sms'] + info['residual_recon']
        audio_full = decode_z_flat_to_audio(dcae, z_full, device)
        save_wav(audio_full, sample_dir / "02_sms_plus_all_atoms.wav")

        cos_full = TF.cosine_similarity(info['z_flat'], z_full, dim=-1).mean().item()
        print(f"    SMS+all cos_sim: {cos_full:.4f}")

        # 4. SMS + each top atom individually
        activations = info['activations'].squeeze(0)  # [T, K]

        for rank, aid in enumerate(top_atom_ids):
            # This atom's contribution: activation * atom_vector
            atom_act = activations[:, aid].unsqueeze(-1)  # [T, 1]

            # Denormalize: the activations were computed on normalized residual,
            # and atoms are unit-normalized. The contribution in original space:
            atom_vec = atoms[aid].unsqueeze(0)  # [1, 128]
            # contribution = activation * atom (in normalized space), then denormalize
            contribution_norm = atom_act * atom_vec  # [T, 128]
            contribution = contribution_norm * res_std.squeeze(0) # scale back

            z_sms_plus_atom = info['z_sms'] + contribution.unsqueeze(0)
            audio_atom = decode_z_flat_to_audio(dcae, z_sms_plus_atom, device)
            save_wav(audio_atom, sample_dir / f"atom_{rank:02d}_id{aid}.wav")

            mean_act = atom_act.mean().item()
            pct_active = (atom_act.squeeze(-1) > 0.01).float().mean().item() * 100
            print(f"    Atom {aid} (rank {rank}): mean_act={mean_act:.3f}, "
                  f"active in {pct_active:.0f}% of frames")

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Test 2: Atom isolation — what does each atom DO?
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Atom Isolation")
    print("  Take a sample's z_sms, add ONLY one atom at varying strengths")
    print("  Hear what each atom adds/changes")
    print("=" * 60)

    iso_dir = OUTPUT_DIR / "atom_isolation"
    iso_dir.mkdir(parents=True, exist_ok=True)

    # Use first test sample as base
    base_sample = test_samples[0]
    base_info = compute_atom_activations(codec, dict_model, base_sample, res_mean, res_std, device)
    z_sms_base = base_info['z_sms']  # [1, T, 128]

    # Save base SMS
    audio_base = decode_z_flat_to_audio(dcae, z_sms_base, device)
    save_wav(audio_base, iso_dir / "00_base_sms.wav")
    print(f"\n  Base sample: {base_sample['name'][:30]}")

    strengths = [0.5, 1.0, 2.0, 4.0]

    for rank, aid in enumerate(top_atom_ids):
        atom_vec = atoms[aid]  # [128]
        # Scale atom to match typical residual magnitude
        atom_scaled = atom_vec * res_std.squeeze(0)  # denormalized direction

        for strength in strengths:
            # Add atom uniformly across all frames
            contribution = atom_scaled.unsqueeze(0).unsqueeze(0) * strength  # [1, 1, 128]
            z_with_atom = z_sms_base + contribution.expand_as(z_sms_base)

            audio = decode_z_flat_to_audio(dcae, z_with_atom, device)
            save_wav(audio, iso_dir / f"atom_{rank:02d}_id{aid}_str{strength:.1f}.wav")

        print(f"  Atom {aid} (rank {rank}): saved 4 strengths (0.5, 1.0, 2.0, 4.0)")

    # ============================================================
    # Test 3: Atom subtraction — remove atoms to hear what's lost
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Atom Subtraction")
    print("  Start from full reconstruction, remove one atom at a time")
    print("  Hear what each atom was contributing")
    print("=" * 60)

    sub_dir = OUTPUT_DIR / "atom_subtraction"
    sub_dir.mkdir(parents=True, exist_ok=True)

    for si in range(min(3, N_SAMPLES)):
        sample = test_samples[si]
        sample_name = sample['name'][:30]
        info = compute_atom_activations(codec, dict_model, sample, res_mean, res_std, device)

        z_full = info['z_sms'] + info['residual_recon']
        audio_full = decode_z_flat_to_audio(dcae, z_full, device)
        save_wav(audio_full, sub_dir / f"sample{si}_full.wav")

        activations = info['activations'].squeeze(0)  # [T, K]

        for rank, aid in enumerate(top_atom_ids[:5]):  # top 5 only
            atom_act = activations[:, aid].unsqueeze(-1)  # [T, 1]
            atom_vec = atoms[aid].unsqueeze(0)  # [1, 128]
            contribution = (atom_act * atom_vec * res_std.squeeze(0)).unsqueeze(0)

            # Full reconstruction MINUS this atom
            z_minus = z_full - contribution
            audio_minus = decode_z_flat_to_audio(dcae, z_minus, device)
            save_wav(audio_minus, sub_dir / f"sample{si}_minus_atom{rank:02d}_id{aid}.wav")

        print(f"  Sample {si} ({sample_name}): saved full + 5 subtractions")

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("ATOM LISTENING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {OUTPUT_DIR}")
    print(f"\nHow to listen:")
    print(f"  sample_*/00_ground_truth.wav  — the real audio")
    print(f"  sample_*/01_sms_only.wav      — what SMS captures (harmonics + noise)")
    print(f"  sample_*/02_sms_plus_all.wav  — SMS + dictionary (full reconstruction)")
    print(f"  sample_*/atom_XX_idYY.wav     — SMS + only atom YY")
    print(f"")
    print(f"  atom_isolation/atom_XX_str*.wav — single atom at varying strengths")
    print(f"  atom_subtraction/sample*_minus_atom*.wav — full minus one atom")
    print(f"")
    print(f"What to listen for:")
    print(f"  - Compare 00 vs 01: what's MISSING from SMS (the residual)")
    print(f"  - Compare 01 vs atom_XX: what each atom ADDS back")
    print(f"  - atom_isolation: hear each atom in isolation at different levels")
    print(f"  - atom_subtraction: hear what's LOST when you remove one atom")


if __name__ == "__main__":
    main()
