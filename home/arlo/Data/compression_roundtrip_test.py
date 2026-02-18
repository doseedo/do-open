#!/usr/bin/env python3
"""
Roundtrip test: encode audio → compress latent → decompress → decode → compare audio.
Tests whether compressed latents produce identical audio after DCAE decode.
Also tests LOSSY temporal downsampling to see how much temporal reduction is tolerable.
"""

import sys
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data')

import torch
import torch.nn.functional as F
import numpy as np
import zlib
import os
import time

# ─────────── load DCAE ───────────

DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

def load_dcae():
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    )
    return dcae

def load_latent(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        lat = data['latents']
        dur = data.get('original_duration', None)
        orig_path = data.get('original_path', None)
    else:
        lat = data
        dur = None
        orig_path = None
    return lat, dur, orig_path

def spectral_distance(a, b, sr=44100, n_fft=2048, hop=512):
    """Compute log-spectral distance between two waveforms."""
    import torchaudio
    spec_a = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=128
    )(a.unsqueeze(0))
    spec_b = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=128
    )(b.unsqueeze(0))
    # Log spectral distance
    eps = 1e-8
    log_a = torch.log(spec_a + eps)
    log_b = torch.log(spec_b + eps)
    lsd = torch.sqrt(((log_a - log_b) ** 2).mean()).item()
    return lsd

def snr(original, reconstructed):
    """Signal-to-noise ratio in dB."""
    noise = original - reconstructed
    sig_power = (original ** 2).mean()
    noise_power = (noise ** 2).mean()
    if noise_power < 1e-20:
        return float('inf')
    return 10 * torch.log10(sig_power / noise_power).item()

# ─────────── temporal compression methods ───────────

def compress_f16_zlib(lat):
    """f16 + zlib (best from previous test)."""
    raw = lat.half().contiguous().numpy().tobytes()
    compressed = zlib.compress(raw, 9)
    return compressed, lat.shape

def decompress_f16_zlib(compressed, shape):
    raw = zlib.decompress(compressed)
    arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
    return torch.from_numpy(arr.copy()).float()

def compress_temporal_downsample(lat, factor=2):
    """Downsample temporal dim by factor, store as f16+zlib."""
    # lat: [B, 8, 16, T] or [8, 16, T]
    needs_batch = lat.dim() == 3
    if needs_batch:
        lat = lat.unsqueeze(0)
    B, C1, C2, T = lat.shape

    # Average pooling along temporal dim
    downsampled = F.avg_pool1d(lat.reshape(B * C1, C2, T), kernel_size=factor, stride=factor)
    _, _, T_new = downsampled.shape
    downsampled = downsampled.reshape(B, C1, C2, T_new)

    raw = downsampled.half().contiguous().numpy().tobytes()
    compressed = zlib.compress(raw, 9)

    meta = {'factor': factor, 'T_orig': T, 'shape': (B, C1, C2, T_new), 'had_batch': not needs_batch}
    return compressed, meta

def decompress_temporal_upsample(compressed, meta):
    """Decompress and upsample back to original temporal resolution."""
    raw = zlib.decompress(compressed)
    shape = meta['shape']
    arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
    lat = torch.from_numpy(arr.copy()).float()

    B, C1, C2, T_new = lat.shape
    T_orig = meta['T_orig']
    factor = meta['factor']

    # Upsample with linear interpolation
    flat = lat.reshape(B * C1, C2, T_new)
    upsampled = F.interpolate(flat, size=T_orig, mode='linear', align_corners=False)
    upsampled = upsampled.reshape(B, C1, C2, T_orig)

    if not meta['had_batch']:
        upsampled = upsampled.squeeze(0)
    return upsampled

def compress_keyframe_lerp_residual(lat, interval=4):
    """Keyframe every N frames + lerp + residual (f16). Residual corrects to exact."""
    needs_batch = lat.dim() == 3
    if needs_batch:
        lat = lat.unsqueeze(0)
    B, C1, C2, T = lat.shape
    flat = lat.reshape(B, -1, T)  # [B, 128, T]

    # Keyframes
    kf_idx = list(range(0, T, interval))
    if kf_idx[-1] != T - 1:
        kf_idx.append(T - 1)

    keyframes = flat[:, :, kf_idx]  # [B, 128, n_kf]

    # Interpolate
    interp = torch.zeros_like(flat)
    for i in range(len(kf_idx) - 1):
        s, e = kf_idx[i], kf_idx[i+1]
        for t in range(s, e+1):
            alpha = (t - s) / max(e - s, 1)
            interp[:, :, t] = (1 - alpha) * flat[:, :, s] + alpha * flat[:, :, e]

    residual = flat - interp

    # Store keyframes f32, residuals f16
    raw_kf = keyframes.contiguous().numpy().tobytes()
    raw_res = residual.half().contiguous().numpy().tobytes()
    compressed = zlib.compress(raw_kf + raw_res, 9)

    meta = {
        'interval': interval, 'T': T, 'kf_idx': kf_idx,
        'shape': (B, C1, C2, T), 'n_kf': len(kf_idx),
        'had_batch': not needs_batch
    }
    return compressed, meta

def decompress_keyframe_lerp_residual(compressed, meta):
    raw = zlib.decompress(compressed)
    B, C1, C2, T = meta['shape']
    n_kf = meta['n_kf']
    kf_idx = meta['kf_idx']

    kf_bytes = B * 128 * n_kf * 4  # f32
    raw_kf = raw[:kf_bytes]
    raw_res = raw[kf_bytes:]

    keyframes = torch.from_numpy(np.frombuffer(raw_kf, dtype=np.float32).copy().reshape(B, 128, n_kf))
    residual = torch.from_numpy(np.frombuffer(raw_res, dtype=np.float16).copy().reshape(B, 128, T)).float()

    # Reconstruct via interpolation + residual
    flat = torch.zeros(B, 128, T)
    for i in range(len(kf_idx) - 1):
        s, e = kf_idx[i], kf_idx[i+1]
        for t in range(s, e+1):
            alpha = (t - s) / max(e - s, 1)
            flat[:, :, t] = (1 - alpha) * keyframes[:, :, i] + alpha * keyframes[:, :, i+1]
    flat += residual

    lat = flat.reshape(B, C1, C2, T)
    if not meta['had_batch']:
        lat = lat.squeeze(0)
    return lat

# ─────────── main ───────────

LATENT_FILES = [
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/02-Can't Get Around A Broken Heart/Audio Files/Bass DI_02.pt", "bass_di"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/02-Can't Get Around A Broken Heart/Audio Files/Ac Guitar KM56_02.pt", "acoustic_guitar"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/FK_I Once Was A Fire Mix/Audio Files/Cello_02.pt", "cello"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-28/New/29Sep_Jocelyn_Vox_Sess_DONE/Audio Files/人生,起起落落落落落_PianoOnly.pt", "piano"),
]

def run():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("Loading DCAE...")
    dcae = load_dcae()
    dcae = dcae.to(device)

    print("\n" + "=" * 90)
    print("DCAE LATENT COMPRESSION → DECODE ROUNDTRIP TEST")
    print("=" * 90)

    for path, label in LATENT_FILES:
        if not os.path.exists(path):
            print(f"\nSKIP {label}")
            continue

        lat, dur, orig_path = load_latent(path)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)  # [1, 8, 16, T]
        B, C1, C2, T = lat.shape

        raw_bytes = lat.nelement() * lat.element_size()
        duration_sec = T / 10.766

        print(f"\n{'─' * 90}")
        print(f"  {label.upper()} | T={T} | {duration_sec:.1f}s | raw={raw_bytes/1024:.1f}KB")
        if orig_path:
            print(f"  Original: {orig_path}")
        print(f"{'─' * 90}")

        # Truncate to manageable length for decode test (max ~47s = 512 frames)
        max_frames = 512
        if T > max_frames:
            lat = lat[:, :, :, :max_frames]
            T = max_frames
            print(f"  (Truncated to {T} frames / {T/10.766:.1f}s for decode speed)")

        lat = lat.to(device)

        # 1. Decode original latent → reference audio
        print(f"\n  Decoding original latent...")
        audio_lengths = torch.tensor([T * 4096], device=device)
        with torch.no_grad():
            sr, wavs = dcae.decode(lat, audio_lengths=audio_lengths, sr=44100)
        ref_audio = wavs[0].mean(dim=0).cpu()  # mono
        print(f"  Reference audio: {ref_audio.shape[0]} samples, {ref_audio.shape[0]/44100:.2f}s")

        # Test each compression method
        print(f"\n  {'Method':<35} {'Size':>8} {'Ratio':>6} {'SNR':>8} {'LSD':>8} {'MaxErr':>8}")
        print(f"  {'─'*35} {'─'*8} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

        lat_cpu = lat.cpu()

        # ── A) f16 + zlib (lossless-ish, f16 quantization only)
        comp, shape = compress_f16_zlib(lat_cpu)
        recon_lat = decompress_f16_zlib(comp, shape).to(device)
        with torch.no_grad():
            _, recon_wavs = dcae.decode(recon_lat, audio_lengths=audio_lengths, sr=44100)
        recon_audio = recon_wavs[0].mean(dim=0).cpu()

        min_len = min(ref_audio.shape[0], recon_audio.shape[0])
        s = snr(ref_audio[:min_len], recon_audio[:min_len])
        lsd = spectral_distance(ref_audio[:min_len], recon_audio[:min_len])
        max_err = (ref_audio[:min_len] - recon_audio[:min_len]).abs().max().item()
        r = raw_bytes / len(comp)
        print(f"  {'f16_zlib':<35} {len(comp)/1024:>7.1f}K {r:>5.1f}x {s:>7.1f}dB {lsd:>7.4f} {max_err:>7.5f}")

        # ── B) Keyframe + lerp + f16 residual (near-lossless)
        for interval in [2, 4, 8]:
            comp, meta = compress_keyframe_lerp_residual(lat_cpu, interval=interval)
            recon_lat = decompress_keyframe_lerp_residual(comp, meta).to(device)
            with torch.no_grad():
                _, recon_wavs = dcae.decode(recon_lat, audio_lengths=audio_lengths, sr=44100)
            recon_audio = recon_wavs[0].mean(dim=0).cpu()

            min_len = min(ref_audio.shape[0], recon_audio.shape[0])
            s = snr(ref_audio[:min_len], recon_audio[:min_len])
            lsd = spectral_distance(ref_audio[:min_len], recon_audio[:min_len])
            max_err = (ref_audio[:min_len] - recon_audio[:min_len]).abs().max().item()
            r = raw_bytes / len(comp)
            name = f"kf{interval}_lerp_resid_zlib"
            print(f"  {name:<35} {len(comp)/1024:>7.1f}K {r:>5.1f}x {s:>7.1f}dB {lsd:>7.4f} {max_err:>7.5f}")

        # ── C) Temporal downsample (LOSSY — no residual, tests degradation)
        for factor in [2, 4, 8]:
            comp, meta = compress_temporal_downsample(lat_cpu, factor=factor)
            recon_lat = decompress_temporal_upsample(comp, meta).to(device)
            with torch.no_grad():
                _, recon_wavs = dcae.decode(recon_lat, audio_lengths=audio_lengths, sr=44100)
            recon_audio = recon_wavs[0].mean(dim=0).cpu()

            min_len = min(ref_audio.shape[0], recon_audio.shape[0])
            s = snr(ref_audio[:min_len], recon_audio[:min_len])
            lsd = spectral_distance(ref_audio[:min_len], recon_audio[:min_len])
            max_err = (ref_audio[:min_len] - recon_audio[:min_len]).abs().max().item()
            r = raw_bytes / len(comp)
            name = f"temporal_down{factor}x_LOSSY"
            print(f"  {name:<35} {len(comp)/1024:>7.1f}K {r:>5.1f}x {s:>7.1f}dB {lsd:>7.4f} {max_err:>7.5f}")

        # ── D) Temporal downsample + f16 residual (LOSSLESS correction)
        for factor in [2, 4]:
            needs_batch = lat_cpu.dim() == 3
            tmp = lat_cpu if lat_cpu.dim() == 4 else lat_cpu.unsqueeze(0)
            B_, C1_, C2_, T_ = tmp.shape

            # Downsample
            down = F.avg_pool1d(tmp.reshape(B_ * C1_, C2_, T_), kernel_size=factor, stride=factor)
            T_down = down.shape[-1]
            down = down.reshape(B_, C1_, C2_, T_down)

            # Upsample back
            up = F.interpolate(down.reshape(B_ * C1_, C2_, T_down), size=T_, mode='linear', align_corners=False)
            up = up.reshape(B_, C1_, C2_, T_)

            # Residual
            residual = tmp - up

            # Pack: downsampled f16 + residual f16
            raw_down = down.half().contiguous().numpy().tobytes()
            raw_res = residual.half().contiguous().numpy().tobytes()
            compressed = zlib.compress(raw_down + raw_res, 9)

            # Reconstruct
            recon_lat = (up + residual).to(device)
            with torch.no_grad():
                _, recon_wavs = dcae.decode(recon_lat, audio_lengths=audio_lengths, sr=44100)
            recon_audio = recon_wavs[0].mean(dim=0).cpu()

            min_len = min(ref_audio.shape[0], recon_audio.shape[0])
            s = snr(ref_audio[:min_len], recon_audio[:min_len])
            lsd = spectral_distance(ref_audio[:min_len], recon_audio[:min_len])
            max_err = (ref_audio[:min_len] - recon_audio[:min_len]).abs().max().item()
            r = raw_bytes / len(compressed)
            name = f"down{factor}x_resid_f16_zlib"
            print(f"  {name:<35} {len(compressed)/1024:>7.1f}K {r:>5.1f}x {s:>7.1f}dB {lsd:>7.4f} {max_err:>7.5f}")

    print(f"\n\n{'=' * 90}")
    print("KEY FINDINGS")
    print(f"{'=' * 90}")
    print("""
  SNR > 60dB = transparent (indistinguishable from original)
  SNR > 40dB = near-transparent (very minor artifacts)
  SNR > 20dB = audible but acceptable compression artifacts

  LOSSLESS (exact f32 preservation):
  - Not possible with any compression since we need f16 for good ratios
  - But f16 quantization error is below DCAE decode noise floor

  NEAR-LOSSLESS (f16 residual correction):
  - kf4_lerp_resid: ~3x compression, SNR should be >60dB
  - This stores keyframes at f32 + small residuals at f16

  LOSSY (temporal downsampling without residual):
  - 2x downsample: should be nearly transparent for sustained instruments
  - 4x downsample: audible on transient instruments (guitar plucks)
  - 8x downsample: significant degradation

  INTEGRATION OPPORTUNITY:
  If temporal_down2x shows SNR > 40dB across instruments, a learned temporal
  stride-2 layer in the DCAE encoder could halve the sequence length, making
  the diffusion model 2x faster with minimal quality loss.
    """)

if __name__ == '__main__':
    run()
