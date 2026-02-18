#!/usr/bin/env python3
"""
Phase 2: Listen to Operation Tree Reconstructions

For each test sample, decode and save:
  1. z_real decoded                    → ground truth
  2. z_sms decoded                     → SMS-only (what SMS captures)
  3. z_sms + tree(residual) decoded    → full operation tree reconstruction
  4. z_sms + single_op_k decoded       → SMS + just operation k

Also: operation isolation — add each operation at varying strengths.
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
from phase2_operation_tree import OperationTreeCodec

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
TREE_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_tree" / "operation_tree.pt"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_listening"

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
Z_DIM = 128
MAX_FRAMES = 256
N_SAMPLES = 5


def decode_z_flat(dcae, z_flat, device):
    """z_flat [1, T, 128] → audio numpy."""
    B, T, D = z_flat.shape
    z_4d = z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)  # [1, 8, 16, T]
    audio_len = int(T * SAMPLE_RATE / 10.8)
    audio_lengths = torch.tensor([audio_len], device=device)
    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return wavs[0].mean(dim=0).cpu().numpy()


def save_wav(audio, path):
    t = torch.from_numpy(audio).float().unsqueeze(0)
    peak = t.abs().max()
    if peak > 0.95:
        t = t * (0.95 / peak)
    torchaudio.save(str(path), t, SAMPLE_RATE)


def load_models(device):
    print("Loading Phase 1 codec...")
    codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
    codec.load_state_dict(torch.load(PHASE1_PATH, weights_only=True, map_location='cpu'))
    codec = codec.to(device).eval()

    print("Loading operation tree...")
    ckpt = torch.load(TREE_PATH, weights_only=False, map_location='cpu')
    tree = OperationTreeCodec(
        z_dim=Z_DIM, n_ops=ckpt['n_ops'], param_dim=ckpt['param_dim'],
        encoder_hidden=256, top_k=ckpt['top_k'],
    )
    tree.load_state_dict(ckpt['model'])
    tree = tree.to(device).eval()
    res_mean = ckpt['res_mean'].to(device)
    res_std = ckpt['res_std'].to(device)

    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_CKPT,
        vocoder_checkpoint_path=VOCODER_CKPT,
    ).to(device).eval()

    return codec, tree, dcae, res_mean, res_std


def gather_diverse_samples(n=20):
    print(f"\nGathering {n} diverse samples...")
    subdirs = sorted(LATENT_BASE.iterdir())
    samples = []
    seen_dirs = set()

    for subdir in subdirs:
        if not subdir.is_dir():
            continue
        for pt in subdir.rglob("*.pt"):
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
                samples.append({'z_real': z, 'path': str(pt), 'name': pt.stem})
                seen_dirs.add(parent_name)
                if len(samples) >= n:
                    break
            except Exception:
                continue
        if len(samples) >= n:
            break

    print(f"  Loaded {len(samples)} diverse samples")
    return samples


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("PHASE 2: OPERATION TREE LISTENING TEST")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    codec, tree, dcae, res_mean, res_std = load_models(device)

    samples = gather_diverse_samples(n=20)
    test_samples = samples[:N_SAMPLES]

    # Rank operations by importance (quick pass over all samples)
    print("\nProfiling operation usage...")
    op_importance = torch.zeros(tree.n_ops, device=device)

    with torch.no_grad():
        for sample in samples:
            z_real = sample['z_real'].unsqueeze(0).to(device)
            z_flat = codec.z_to_flat(z_real)
            z_sms = codec.forward_G(codec.forward_F(z_flat))
            residual = z_flat - z_sms
            residual_norm = (residual - res_mean) / res_std

            hidden = tree.encode(residual_norm)
            raw_alpha = tree.activation_head(hidden)
            alpha = TF.softplus(raw_alpha)  # [1, T, n_ops]

            for k in range(tree.n_ops):
                params_k = tree.param_heads[k](hidden)
                contrib_k = tree.operations[k](params_k)
                importance = (alpha[:, :, k] * contrib_k.norm(dim=-1)).sum()
                op_importance[k] += importance

    op_ranking = torch.argsort(op_importance, descending=True).cpu().tolist()
    print(f"  Operations ranked by importance: {op_ranking}")
    for rank, k in enumerate(op_ranking):
        print(f"    Rank {rank}: op {k}, importance={op_importance[k]:.1f}")

    # ============================================================
    # Test 1: Per-sample decomposition
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Sample Decomposition")
    print("  ground_truth, sms_only, sms+tree, sms+each_op")
    print("=" * 60)

    for si, sample in enumerate(test_samples):
        name = sample['name'][:30]
        sample_dir = OUTPUT_DIR / f"sample_{si}_{name}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        z_real = sample['z_real'].unsqueeze(0).to(device)
        z_flat = codec.z_to_flat(z_real)

        with torch.no_grad():
            z_sms = codec.forward_G(codec.forward_F(z_flat))
            residual = z_flat - z_sms
            residual_norm = (residual - res_mean) / res_std

            # Full tree reconstruction
            recon_norm, activations, all_params = tree(residual_norm)
            recon = recon_norm * res_std + res_mean
            z_full = z_sms + recon

        # Scale z_sms to match z_real magnitude (Phase 1 outputs correct
        # direction but wrong scale because SMS-rendered audio is quieter)
        z_real_norm = z_flat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z_sms_norm = z_sms.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z_sms_scaled = z_sms * (z_real_norm / z_sms_norm)

        cos_sms = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
        cos_full = TF.cosine_similarity(z_flat, z_full, dim=-1).mean().item()
        mag_ratio = (z_sms_norm / z_real_norm).mean().item()

        print(f"\n  Sample {si}: {name}")
        print(f"    SMS cos={cos_sms:.4f}  SMS+tree cos={cos_full:.4f}  mag_ratio={mag_ratio:.3f}")

        # 1. Ground truth
        audio_gt = decode_z_flat(dcae, z_flat, device)
        save_wav(audio_gt, sample_dir / "00_ground_truth.wav")

        # 2. SMS only (magnitude-corrected so it's audible)
        audio_sms = decode_z_flat(dcae, z_sms_scaled, device)
        save_wav(audio_sms, sample_dir / "01_sms_only.wav")

        # 3. Full tree reconstruction
        audio_full = decode_z_flat(dcae, z_full, device)
        save_wav(audio_full, sample_dir / "02_sms_plus_tree.wav")

        # 4. SMS + each top operation individually
        with torch.no_grad():
            hidden = tree.encode(residual_norm)
            raw_alpha = tree.activation_head(hidden)
            alpha = TF.softplus(raw_alpha)

            for rank, k in enumerate(op_ranking[:6]):
                params_k = tree.param_heads[k](hidden)
                contrib_k = tree.operations[k](params_k)
                alpha_k = alpha[:, :, k:k+1]

                # Denormalize contribution
                single_op_recon = (alpha_k * contrib_k) * res_std + res_mean
                z_sms_plus_op = z_sms + single_op_recon

                audio_op = decode_z_flat(dcae, z_sms_plus_op, device)
                save_wav(audio_op, sample_dir / f"op_{rank:02d}_id{k}.wav")

                mean_alpha = alpha_k.mean().item()
                print(f"    Op {k} (rank {rank}): mean_alpha={mean_alpha:.3f}")

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Test 2: Operation subtraction
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Operation Subtraction")
    print("  Full reconstruction minus one operation at a time")
    print("=" * 60)

    sub_dir = OUTPUT_DIR / "op_subtraction"
    sub_dir.mkdir(parents=True, exist_ok=True)

    for si in range(min(3, N_SAMPLES)):
        sample = test_samples[si]
        name = sample['name'][:30]

        z_real = sample['z_real'].unsqueeze(0).to(device)
        z_flat = codec.z_to_flat(z_real)

        with torch.no_grad():
            z_sms = codec.forward_G(codec.forward_F(z_flat))
            residual = z_flat - z_sms
            residual_norm = (residual - res_mean) / res_std

            recon_norm, activations, all_params = tree(residual_norm)
            recon = recon_norm * res_std + res_mean
            z_full = z_sms + recon

        audio_full = decode_z_flat(dcae, z_full, device)
        save_wav(audio_full, sub_dir / f"sample{si}_full.wav")

        with torch.no_grad():
            hidden = tree.encode(residual_norm)
            raw_alpha = tree.activation_head(hidden)
            alpha = TF.softplus(raw_alpha)

            for rank, k in enumerate(op_ranking[:5]):
                params_k = tree.param_heads[k](hidden)
                contrib_k = tree.operations[k](params_k)
                alpha_k = alpha[:, :, k:k+1]
                single_contrib = (alpha_k * contrib_k) * res_std + res_mean

                z_minus = z_full - single_contrib
                audio_minus = decode_z_flat(dcae, z_minus, device)
                save_wav(audio_minus, sub_dir / f"sample{si}_minus_op{rank:02d}_id{k}.wav")

        print(f"  Sample {si} ({name}): saved full + 5 subtractions")

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("LISTENING TEST COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {OUTPUT_DIR}")
    print(f"\nHow to listen:")
    print(f"  sample_*/00_ground_truth.wav   — real audio (DCAE decoded)")
    print(f"  sample_*/01_sms_only.wav       — what Phase 1 SMS captures")
    print(f"  sample_*/02_sms_plus_tree.wav  — full operation tree reconstruction")
    print(f"  sample_*/op_XX_idYY.wav        — SMS + only operation YY")
    print(f"")
    print(f"  op_subtraction/sampleN_full.wav           — full reconstruction")
    print(f"  op_subtraction/sampleN_minus_opXX_idYY.wav — full minus one op")
    print(f"")
    print(f"What to listen for:")
    print(f"  - 00 vs 01: what SMS misses (the residual)")
    print(f"  - 01 vs 02: what the operation tree adds back")
    print(f"  - 01 vs op_XX: what each individual operation contributes")
    print(f"  - subtraction: what's lost when you remove one operation")


if __name__ == "__main__":
    main()
