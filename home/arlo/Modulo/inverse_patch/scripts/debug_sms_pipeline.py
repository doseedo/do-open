#!/usr/bin/env python3
"""
Debug: Compare SMS at every stage of the pipeline.

Key discovery: SMS .pt files were extracted at DCAE's frame rate (~10.8 fps,
hop≈4085 at 44.1kHz), NOT at hop=512. The T dimension in SMS matches T in
DCAE latents exactly. So additive_synth must use the correct hop.

For paired samples (SMS .pt + real audio latent + original wav):
  A. original_wav   = load original .wav from GCS              (ground truth)
  B. sms_raw        = SMS .pt → additive_synth (correct hop)   (SMS reconstruction)
  C. sms_thru_dcae  = SMS → synth → DCAE encode → decode       (SMS through DCAE)
  D. neural_roundtrip = z_real → F → G → DCAE decode           (what Phase 2 calls "SMS only")
"""

import sys
import torch
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
import torchaudio
import orjson

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM, additive_synth
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
SMS_BASE = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "debug_sms_pipeline"

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

TARGET_SR = 44100
N_SAMPLES = 8


def save_wav(audio_np, path, sr=TARGET_SR):
    t = torch.from_numpy(audio_np).float().unsqueeze(0)
    peak = t.abs().max()
    if peak > 0.95:
        t = t * (0.95 / peak)
    torchaudio.save(str(path), t, sr)


def decode_z(dcae, z_4d, n_audio_samples, device):
    """Decode z [1, 8, 16, T] → audio, using correct audio length."""
    audio_lengths = torch.tensor([n_audio_samples], device=device)
    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=TARGET_SR)
    return wavs[0].mean(dim=0).cpu().numpy()


def encode_audio(dcae, audio_np, device):
    """Encode audio → z [1, 8, 16, T]."""
    audio_t = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)
    audio_t = audio_t.expand(-1, 2, -1).to(device)
    audio_lengths = torch.tensor([audio_t.shape[-1]], device=device)
    with torch.no_grad():
        z, _ = dcae.encode(audio_t, audio_lengths=audio_lengths, sr=TARGET_SR)
    return z


def load_original_audio(audio_path, target_sr=TARGET_SR):
    """Load original wav, resample to target_sr if needed, return mono numpy."""
    audio, sr = torchaudio.load(str(audio_path))
    # Mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0).numpy()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("DEBUG: SMS PIPELINE COMPARISON")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print("\nLoading manifest...")
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    # Find paired samples where SMS .pt, latent .pt, AND original wav exist
    paired = []
    for entry in manifest['entries']:
        sms_path = SMS_BASE / entry['path']
        lat_path = Path(entry['latent_path'])
        if not (sms_path.exists() and lat_path.exists()):
            continue

        # Get audio_path from SMS .pt metadata
        sms_data = torch.load(sms_path, weights_only=True, map_location='cpu')
        audio_path = sms_data.get('audio_path', '')
        if isinstance(audio_path, str) and Path(audio_path).exists():
            paired.append({
                'sms_path': sms_path,
                'latent_path': lat_path,
                'audio_path': Path(audio_path),
                'name': lat_path.stem,
            })
        if len(paired) >= N_SAMPLES * 3:
            break

    print(f"  Found {len(paired)} fully paired samples (SMS + latent + wav)")

    # Filter for diversity
    selected = []
    seen_parents = set()
    for p in paired:
        parent = p['latent_path'].parent.parent.name
        if parent in seen_parents:
            continue
        seen_parents.add(parent)
        selected.append(p)
        if len(selected) >= N_SAMPLES:
            break

    if not selected:
        print("  No paired samples found! Trying without diversity filter...")
        selected = paired[:N_SAMPLES]

    print(f"  Selected {len(selected)} samples")

    # Load DCAE
    print("\nLoading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_CKPT,
        vocoder_checkpoint_path=VOCODER_CKPT,
    ).to(device).eval()

    # Load Phase 1 codec
    print("Loading Phase 1 codec...")
    codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=128, g_hidden=384, f_hidden=256)
    codec.load_state_dict(torch.load(PHASE1_PATH, weights_only=True, map_location='cpu'))
    codec = codec.to(device).eval()

    print("\n" + "=" * 60)
    print("GENERATING COMPARISONS")
    print("=" * 60)

    for i, sample in enumerate(selected):
        name = sample['name'][:40]
        print(f"\n{'─'*60}")
        print(f"  Sample {i}: {name}")
        print(f"  Audio: {sample['audio_path']}")

        sample_dir = OUTPUT_DIR / f"sample_{i:02d}_{name}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # --- Load SMS .pt ---
        sms_data = torch.load(sample['sms_path'], weights_only=True, map_location='cpu')
        freqs = sms_data['freqs']       # [T, 128]
        amps = sms_data['amps']         # [T, 128]
        noise = sms_data.get('noise_amps')  # [T, 8]
        T_sms = freqs.shape[0]

        # --- Load z_real ---
        lat_data = torch.load(sample['latent_path'], weights_only=False, map_location='cpu')
        z_real = lat_data['latents']  # [8, 16, T]
        T_z = z_real.shape[2]
        original_duration = lat_data.get('original_duration', T_z * 4085 / TARGET_SR)

        # --- Load original audio ---
        audio_original = load_original_audio(sample['audio_path'])
        n_audio_samples = len(audio_original)

        # Compute correct hop for SMS (SMS T matches DCAE T)
        sms_hop = n_audio_samples // T_sms
        sms_duration = n_audio_samples / TARGET_SR

        print(f"    T_sms={T_sms}, T_z={T_z}, duration={sms_duration:.2f}s")
        print(f"    SMS hop={sms_hop} samples ({sms_hop/TARGET_SR*1000:.1f}ms)")
        print(f"    Audio samples: {n_audio_samples}")

        # ============================================================
        # A. Original audio (from wav file)
        # ============================================================
        save_wav(audio_original, sample_dir / "A_original_wav.wav")
        print(f"    A) Original wav: {sms_duration:.2f}s")

        # ============================================================
        # A2. z_real decoded through DCAE
        # ============================================================
        z_real_4d = z_real.unsqueeze(0).to(device)
        audio_z_decoded = decode_z(dcae, z_real_4d, n_audio_samples, device)
        save_wav(audio_z_decoded, sample_dir / "A2_z_real_decoded.wav")
        print(f"    A2) z_real → DCAE decode: {len(audio_z_decoded)/TARGET_SR:.2f}s")

        # ============================================================
        # B. SMS Raw: additive synthesis with CORRECT hop
        # ============================================================
        audio_sms_raw = additive_synth(freqs, amps, noise, sr=TARGET_SR, hop=sms_hop)
        peak = np.abs(audio_sms_raw).max()
        if peak > 0:
            audio_sms_raw = audio_sms_raw / peak * 0.8
        save_wav(audio_sms_raw, sample_dir / "B_sms_raw.wav")
        print(f"    B) SMS raw (hop={sms_hop}): {len(audio_sms_raw)/TARGET_SR:.2f}s")

        # ============================================================
        # C. SMS through DCAE: synth → DCAE encode → decode
        # ============================================================
        z_sms_encoded = encode_audio(dcae, audio_sms_raw, device)
        audio_sms_dcae = decode_z(dcae, z_sms_encoded, len(audio_sms_raw), device)
        save_wav(audio_sms_dcae, sample_dir / "C_sms_thru_dcae.wav")

        T_sms_enc = z_sms_encoded.shape[-1]
        print(f"    C) SMS → DCAE encode (T={T_sms_enc}) → decode: "
              f"{len(audio_sms_dcae)/TARGET_SR:.2f}s")

        # Cos sim z_real vs z_sms_encoded
        T_min = min(T_z, T_sms_enc)
        cos_real_sms = TF.cosine_similarity(
            z_real_4d[:,:,:,:T_min].reshape(1, -1),
            z_sms_encoded[:,:,:,:T_min].reshape(1, -1)
        ).item()
        print(f"       cos(z_real, z_sms_dcae)={cos_real_sms:.4f}")

        # ============================================================
        # D. Neural roundtrip: z_real → F → G → DCAE decode
        # ============================================================
        z_flat = codec.z_to_flat(z_real_4d)
        with torch.no_grad():
            sms_pred = codec.forward_F(z_flat)
            z_roundtrip = codec.forward_G(sms_pred)

        z_rt_4d = codec.flat_to_z(z_roundtrip)
        audio_roundtrip = decode_z(dcae, z_rt_4d, n_audio_samples, device)
        save_wav(audio_roundtrip, sample_dir / "D_neural_roundtrip.wav")

        cos_rt = TF.cosine_similarity(
            z_flat.reshape(1, -1), z_roundtrip.reshape(1, -1)
        ).item()
        print(f"    D) z→F→G→decode: {len(audio_roundtrip)/TARGET_SR:.2f}s  "
              f"cos={cos_rt:.4f}")

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n    Compare:")
        print(f"      A vs A2: DCAE encode→decode fidelity on real audio")
        print(f"      A vs B:  what SMS captures vs original")
        print(f"      B vs C:  what DCAE does to SMS audio")
        print(f"      A2 vs D: neural codec error (cos={cos_rt:.4f})")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {OUTPUT_DIR}")
    print(f"\n  A_original_wav.wav       — original audio file")
    print(f"  A2_z_real_decoded.wav    — z_real → DCAE decode")
    print(f"  B_sms_raw.wav            — SMS → additive synth (correct hop)")
    print(f"  C_sms_thru_dcae.wav      — SMS → synth → DCAE → decode")
    print(f"  D_neural_roundtrip.wav   — z_real → F → G → DCAE decode")


if __name__ == "__main__":
    main()
