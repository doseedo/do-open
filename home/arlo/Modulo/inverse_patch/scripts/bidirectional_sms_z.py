#!/usr/bin/env python3
"""
Phase 1: Bidirectional SMS-Z Codec

Train a cycle-consistent bidirectional mapping between:
  - Compressed SMS params (102 dims/frame)
  - DCAE z of SMS-RENDERED audio (128 dims/frame)

Key insight: SMS fully determines the rendered audio, which fully determines z_sms.
There is ZERO information gap, so the mapping should be perfectly invertible.

Pipeline:
  1. Load SMS .pt files
  2. Render to audio via additive synth
  3. Encode rendered audio with DCAE → z_sms
  4. Compress SMS → hierarchical program (102 dims)
  5. Train G(sms→z) and F(z→sms) with cycle consistency
  6. Test roundtrip reconstruction
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import torchaudio
import orjson
import os

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
HOP_SIZE = 512  # Legacy — only used as fallback
# DCAE actual hop ≈ 4085 samples at 44.1kHz (~10.8 fps).
# SMS .pt T matches DCAE T exactly, so correct hop = n_audio_samples / T.

# Compression params
USE_COMPRESSION = True  # False = raw 128 sines + noise, True = harmonic groups + indep
MAX_GROUPS = 6
MAX_PARTIALS = 8
MAX_INDEPENDENT = 20
N_NOISE_BANDS = 8
N_RAW_SINES = 128

if USE_COMPRESSION:
    SMS_DIM = MAX_GROUPS + MAX_GROUPS * MAX_PARTIALS + MAX_INDEPENDENT * 2 + N_NOISE_BANDS  # 102
else:
    SMS_DIM = N_RAW_SINES * 2 + N_NOISE_BANDS  # 264 (128 freqs + 128 amps + 8 noise)

LOG10_20 = np.log10(20)
LOG10_20K = np.log10(20000)


# ============================================================
# Additive Synthesis (SMS params → audio)
# ============================================================

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def get_noise_band_edges(n_bands=N_NOISE_BANDS, min_freq=80, max_freq=16000):
    """Mel-spaced noise band edges."""
    min_mel = hz_to_mel(min_freq)
    max_mel = hz_to_mel(max_freq)
    mel_edges = np.linspace(min_mel, max_mel, n_bands + 1)
    return mel_to_hz(mel_edges)


def additive_synth(freqs, amps, noise_amps=None, sr=SAMPLE_RATE, hop=HOP_SIZE):
    """
    Render SMS params to audio via additive synthesis.

    Args:
        freqs: [T, n_sines] Hz
        amps: [T, n_sines]
        noise_amps: [T, n_noise_bands] or None
        sr: sample rate
        hop: samples per frame

    Returns:
        audio: [n_samples] float32
    """
    T, n_sines = freqs.shape
    n_samples = T * hop
    audio = np.zeros(n_samples, dtype=np.float64)

    freqs_np = freqs.numpy().astype(np.float64)
    amps_np = amps.numpy().astype(np.float64)

    # Time axis for interpolation
    t_samples = np.arange(n_samples, dtype=np.float64) / sr
    t_frames = np.arange(T, dtype=np.float64) * hop / sr

    # Render each active sine track
    for s in range(n_sines):
        a_track = amps_np[:, s]
        if a_track.max() < 0.0005:
            continue

        f_track = freqs_np[:, s]

        # Interpolate freq and amp to sample rate
        f_interp = np.interp(t_samples, t_frames, f_track)
        a_interp = np.interp(t_samples, t_frames, a_track)

        # Phase-continuous synthesis: integrate instantaneous frequency
        phase = np.cumsum(2 * np.pi * f_interp / sr)
        audio += a_interp * np.sin(phase)

    # Render noise bands
    if noise_amps is not None:
        noise_np = noise_amps.numpy().astype(np.float64)
        band_edges = get_noise_band_edges()

        # Compute per-frame sine energy to scale noise relative to signal
        sine_energy = np.sqrt(np.sum(amps_np ** 2, axis=1))  # [T] RMS across partials
        # Interpolate to sample rate
        sine_energy_interp = np.interp(t_samples, t_frames, sine_energy)

        # Generate base white noise (shared, then filtered per band)
        rng = np.random.RandomState(42)  # deterministic for reproducibility
        white = rng.randn(n_samples)

        # FFT-based bandpass filtering
        fft_noise = np.fft.rfft(white)
        fft_freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)

        for b in range(noise_np.shape[1]):
            na_track = noise_np[:, b]
            if na_track.max() < 0.0005:
                continue

            lo = band_edges[b]
            hi = band_edges[b + 1]

            # Bandpass mask
            mask = (fft_freqs >= lo) & (fft_freqs <= hi)
            band_fft = fft_noise.copy()
            band_fft[~mask] = 0

            band_audio = np.fft.irfft(band_fft, n_samples)

            # Normalize band energy
            band_rms = np.sqrt(np.mean(band_audio ** 2)) + 1e-10
            band_audio = band_audio / band_rms

            # Modulate by time-varying amplitude
            a_interp = np.interp(t_samples, t_frames, na_track)

            # Clamp noise: don't let it exceed 2x the local sine energy
            # This prevents noise from dominating in silent/tail regions
            max_noise = np.maximum(sine_energy_interp * 2.0, 0.001)
            a_interp = np.minimum(a_interp, max_noise)

            audio += a_interp * band_audio

    return audio.astype(np.float32)


# ============================================================
# Harmonic Group Detection
# ============================================================

def find_harmonic_groups(freqs, amps, f0_threshold=20, ratio_tolerance=0.05):
    """Find groups of sines at integer frequency ratios."""
    T, n_sines = freqs.shape
    avg_freqs = freqs.mean(dim=0).numpy()
    avg_amps = amps.mean(dim=0).numpy()
    amp_order = np.argsort(avg_amps)[::-1]

    groups = []
    used = set()

    for i in amp_order:
        if i in used or avg_amps[i] < 0.001:
            continue
        f0 = avg_freqs[i]
        if f0 < f0_threshold:
            continue

        group = {'f0': f0, 'partials': [(1, int(i), float(avg_amps[i]))]}
        used.add(int(i))

        for j in range(n_sines):
            if j in used or avg_amps[j] < 0.0001:
                continue
            ratio = avg_freqs[j] / f0
            nearest_int = round(ratio)
            if 2 <= nearest_int <= 16:
                if abs(ratio - nearest_int) / nearest_int < ratio_tolerance:
                    group['partials'].append((nearest_int, int(j), float(avg_amps[j])))
                    used.add(int(j))

        if len(group['partials']) >= 2:
            groups.append(group)

    return groups, used


# ============================================================
# SMS Compression
# ============================================================

def compress_sms(freqs, amps, noise_amps):
    """Compress [T, 128] SMS → 102 dense dims per frame."""
    T, n_sines = freqs.shape
    groups, used_sines = find_harmonic_groups(freqs, amps)

    group_f0s = torch.zeros(T, MAX_GROUPS)
    group_partial_amps = torch.zeros(T, MAX_GROUPS, MAX_PARTIALS)

    for g_idx, group in enumerate(groups[:MAX_GROUPS]):
        fund_idx = group['partials'][0][1]
        group_f0s[:, g_idx] = freqs[:, fund_idx]
        for p_idx, (ratio, sine_idx, _) in enumerate(group['partials'][:MAX_PARTIALS]):
            group_partial_amps[:, g_idx, p_idx] = amps[:, sine_idx]

    remaining = sorted(
        [i for i in range(n_sines) if i not in used_sines],
        key=lambda i: amps[:, i].mean().item(), reverse=True
    )

    indep_freqs = torch.zeros(T, MAX_INDEPENDENT)
    indep_amps = torch.zeros(T, MAX_INDEPENDENT)
    for j, sine_idx in enumerate(remaining[:MAX_INDEPENDENT]):
        indep_freqs[:, j] = freqs[:, sine_idx]
        indep_amps[:, j] = amps[:, sine_idx]

    group_partial_amps = group_partial_amps.view(T, MAX_GROUPS * MAX_PARTIALS)

    return {
        'group_f0s': group_f0s,
        'group_amps': group_partial_amps,
        'indep_freqs': indep_freqs,
        'indep_amps': indep_amps,
        'noise_amps': noise_amps,
    }


def normalize_sms(group_f0s, group_amps, indep_freqs, indep_amps, noise_amps):
    """Normalize SMS params to model input space. Returns [B, T, 102]."""
    log_f0s = torch.log10(group_f0s.clamp(min=20, max=20000))
    log_f0s = (log_f0s - LOG10_20) / (LOG10_20K - LOG10_20)
    log_f0s = log_f0s * (group_f0s > 20).float()

    log_ifreqs = torch.log10(indep_freqs.clamp(min=20, max=20000))
    log_ifreqs = (log_ifreqs - LOG10_20) / (LOG10_20K - LOG10_20)
    log_ifreqs = log_ifreqs * (indep_freqs > 20).float()

    return torch.cat([log_f0s, group_amps, log_ifreqs, indep_amps, noise_amps], dim=-1)


# ============================================================
# Data Preparation: Load SMS, Render, Encode with DCAE
# ============================================================

def _process_single_sms(path):
    """
    CPU worker: load .pt, render additive synth, compress, normalize.
    Returns (audio_np, sms_norm, n_audio_samples, path) or None on failure.
    """
    try:
        sms_data = torch.load(path, weights_only=True, map_location='cpu')
        freqs = sms_data['freqs']
        amps = sms_data['amps']
        noise = sms_data.get('noise_amps')
        if noise is None:
            return None

        T = freqs.shape[0]
        if T < 20:
            return None

        # Compute correct hop from original audio duration.
        # SMS T matches DCAE T — both at DCAE frame rate (~10.8 fps).
        audio_path = sms_data.get('audio_path', '')
        hop = HOP_SIZE  # fallback
        n_audio_samples = T * HOP_SIZE  # fallback

        if isinstance(audio_path, str) and os.path.exists(audio_path):
            try:
                info = torchaudio.info(audio_path)
                orig_sr = info.sample_rate
                orig_samples = info.num_frames
                # Convert to target sample rate
                n_audio_samples = int(orig_samples * SAMPLE_RATE / orig_sr)
                hop = n_audio_samples // T
            except Exception:
                pass  # use fallback

        # Find where content starts (skip leading silence)
        frame_energy = amps.sum(dim=1)  # [T]
        start_frame = 0
        for t in range(T):
            if frame_energy[t] > 0.001:
                start_frame = t
                break
        else:
            return None  # entirely silent SMS

        # Take up to 3 seconds starting from first active frame
        dur_frames = min(256, int(3.0 * SAMPLE_RATE / hop)) if hop > 0 else 256
        end_frame = min(T, start_frame + dur_frames)
        if end_frame - start_frame < 10:
            return None  # not enough active content

        freqs = freqs[start_frame:end_frame]
        amps = amps[start_frame:end_frame]
        noise = noise[start_frame:end_frame]
        T = freqs.shape[0]
        n_audio_samples = T * hop  # recalculate after truncation

        # Render SMS to audio with correct hop
        audio = additive_synth(freqs, amps, noise, sr=SAMPLE_RATE, hop=hop)

        # Skip samples with too-quiet rendered audio
        if np.abs(audio).max() < 0.005:
            return None

        if USE_COMPRESSION:
            # Compress + normalize SMS (CPU)
            program = compress_sms(freqs, amps, noise)
            sms_norm = normalize_sms(
                program['group_f0s'],
                program['group_amps'],
                program['indep_freqs'],
                program['indep_amps'],
                program['noise_amps'],
            )
        else:
            # Raw 128 sines + noise: normalize freqs to log [0,1], pass amps + noise as-is
            log_freqs = torch.log10(freqs.clamp(min=20, max=20000))
            log_freqs = (log_freqs - LOG10_20) / (LOG10_20K - LOG10_20)
            log_freqs = log_freqs * (freqs > 20).float()
            sms_norm = torch.cat([log_freqs, amps, noise], dim=-1)  # [T, 264]

        return (audio, sms_norm, n_audio_samples, path)
    except Exception:
        return None


def prepare_cpu(manifest_path, max_samples=800):
    """
    CPU phase: load SMS, render additive synth, compress, normalize.
    Runs BEFORE DCAE is loaded.
    """
    print(f"\nLoading SMS data from {manifest_path}...")
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    # Resolve relative paths against manifest's parent directory
    manifest_dir = Path(manifest_path).parent.parent  # data/sms_v4 → inverse_patch
    cpu_results = []
    skipped = 0

    for entry in manifest['entries']:
        if len(cpu_results) >= max_samples:
            break

        path = entry['path']
        if any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            continue
        # Try as-is first, then resolve relative to manifest base
        if not os.path.exists(path):
            path = str(manifest_dir / path)
        if not os.path.exists(path):
            continue

        result = _process_single_sms(path)
        if result is not None:
            cpu_results.append(result)
            if len(cpu_results) % 50 == 0:
                print(f"    Rendered {len(cpu_results)} samples...")
        else:
            skipped += 1

    print(f"  CPU done: {len(cpu_results)} samples rendered (skipped {skipped})")
    return cpu_results


def prepare_gpu(cpu_results, dcae, device):
    """
    GPU phase: encode each rendered audio with DCAE.
    Called AFTER parallel CPU work is done.
    """
    print(f"  Encoding {len(cpu_results)} samples with DCAE...")
    data = []
    errors = 0

    for i, (audio, sms_norm, n_audio_samples, path) in enumerate(cpu_results):
        try:
            # Debug first sample
            if i == 0:
                print(f"    First sample: audio shape={audio.shape}, "
                      f"min={audio.min():.6f}, max={audio.max():.6f}, "
                      f"rms={np.sqrt(np.mean(audio**2)):.6f}, "
                      f"duration={audio.shape[0]/SAMPLE_RATE:.2f}s")
                print(f"    sms_norm shape={sms_norm.shape}")

            # DCAE expects stereo [B, 2, T]
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            audio_t = audio_t.expand(-1, 2, -1).to(device)  # [1, 2, T]
            audio_lengths = torch.tensor([audio_t.shape[-1]], device=device)

            with torch.no_grad():
                z_sms, _ = dcae.encode(audio_t, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
            z_sms = z_sms.squeeze(0).cpu()  # [8, 16, T_z]

            if i == 0:
                T_z_first = z_sms.shape[-1]
                T_sms_first = sms_norm.shape[0]
                print(f"    First z_sms shape={z_sms.shape}, T_sms={T_sms_first}, T_z={T_z_first}")
                if abs(T_z_first - T_sms_first) > 2:
                    print(f"    WARNING: T mismatch! SMS T={T_sms_first} vs DCAE T={T_z_first}")
                else:
                    print(f"    T match OK (SMS={T_sms_first}, DCAE={T_z_first})")

            T_z = z_sms.shape[-1]
            T_sms = sms_norm.shape[0]
            if T_z < 5:
                continue

            T_min = min(T_sms, T_z)
            sms_norm = sms_norm[:T_min]
            z_sms = z_sms[:, :, :T_min]

            data.append({
                'sms_norm': sms_norm,
                'z_sms': z_sms,
                'n_audio_samples': n_audio_samples,
                'path': path,
            })

            if len(data) % 100 == 0:
                print(f"    Encoded {len(data)} samples...")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    ERROR on sample {i}: {e}")
            continue

    print(f"  Total: {len(data)} samples ready")
    return data


# ============================================================
# Bidirectional Model
# ============================================================

class DirectionModel(nn.Module):
    """One direction of the bidirectional codec."""

    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )

        n_gru_layers = 3 if hidden_dim >= 512 else 2
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=n_gru_layers, batch_first=True,
            bidirectional=True, dropout=0.1
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Default init — MSE loss will drive correct magnitude

    def forward(self, x):
        """
        x: [B, T, input_dim]
        Returns: [B, T, output_dim]
        """
        h = self.encoder(x)          # [B, T, hidden]
        h = h.permute(0, 2, 1)       # [B, hidden, T]
        h = self.temporal(h)
        h = h.permute(0, 2, 1)       # [B, T, hidden]
        h, _ = self.gru(h)           # [B, T, hidden*2]
        return self.output_proj(h)   # [B, T, output_dim]


class BidirectionalCodec(nn.Module):
    """
    Bidirectional SMS ↔ Z codec with cycle consistency.

    G (forward):  SMS_norm [T, 102] → z_flat [T, 128]
    F (reverse):  z_flat [T, 128] → SMS_norm [T, 102]
    """

    def __init__(self, sms_dim=SMS_DIM, z_dim=128, g_hidden=768, f_hidden=256):
        super().__init__()
        self.sms_dim = sms_dim
        self.z_dim = z_dim

        self.G = DirectionModel(sms_dim, z_dim, g_hidden)    # SMS → z (bigger, harder direction)
        self.F = DirectionModel(z_dim, sms_dim, f_hidden)    # z → SMS (smaller, easier direction)

    def forward_G(self, sms_norm):
        """SMS [B, T, 102] → z_flat [B, T, 128]"""
        return self.G(sms_norm)

    def forward_F(self, z_flat):
        """z_flat [B, T, 128] → SMS [B, T, 102]"""
        return self.F(z_flat)

    def z_to_flat(self, z):
        """z [B, 8, 16, T] → z_flat [B, T, 128]"""
        B, C, H, T = z.shape
        return z.permute(0, 3, 1, 2).reshape(B, T, C * H)

    def flat_to_z(self, z_flat):
        """z_flat [B, T, 128] → z [B, 8, 16, T]"""
        B, T, _ = z_flat.shape
        return z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)


# ============================================================
# Training
# ============================================================

def _pad_batch_sms(samples, device):
    """Pad variable-T samples into batched tensors + mask."""
    sms_list = [s['sms_norm'] for s in samples]           # each [T, 102]
    z_list = [s['z_sms'].permute(2, 0, 1).reshape(-1, 128) for s in samples]  # each [T, 128]

    T_max = max(s.shape[0] for s in sms_list)
    B = len(samples)

    sms_pad = torch.zeros(B, T_max, SMS_DIM)
    z_pad = torch.zeros(B, T_max, 128)
    mask = torch.zeros(B, T_max)

    for i in range(B):
        T = sms_list[i].shape[0]
        sms_pad[i, :T] = sms_list[i]
        z_pad[i, :T] = z_list[i][:T]  # ensure same T
        mask[i, :T] = 1.0

    return sms_pad.to(device), z_pad.to(device), mask.to(device)


def train_bidirectional(data, device='cuda', epochs=500, batch_size=32,
                        cycle_weight=1.0):
    """Train with cycle consistency using padded batches."""
    print("\n" + "=" * 60)
    print("TRAINING BIDIRECTIONAL CODEC")
    print("=" * 60)

    z_dim = data[0]['z_sms'].shape[0] * data[0]['z_sms'].shape[1]  # 8*16=128
    print(f"  SMS dim: {SMS_DIM}")
    print(f"  Z dim (flat): {z_dim}")
    print(f"  Batch size: {batch_size}")

    model = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=z_dim, g_hidden=384, f_hidden=256).to(device)
    g_params = sum(p.numel() for p in model.G.parameters())
    f_params = sum(p.numel() for p in model.F.parameters())
    print(f"  G (SMS→z) params: {g_params:,} (hidden=384)")
    print(f"  F (z→SMS) params: {f_params:,} (hidden=256)")
    print(f"  Forward loss: cosine + magnitude  |  Reverse loss: MSE")
    print(f"  Cycle warmup: 0→1 over first 50 epochs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    from tqdm import tqdm

    for epoch in range(epochs):
        total_fwd = 0
        total_rev = 0
        total_cyc_fwd = 0
        total_cyc_rev = 0
        total_samples = 0

        indices = np.random.permutation(len(data))

        pbar = tqdm(range(0, len(data), batch_size),
                    desc=f"Epoch {epoch:3d}", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for i in pbar:
            batch_idx = indices[i:i+batch_size]
            batch = [data[j] for j in batch_idx]
            B = len(batch)

            sms, z_flat, mask = _pad_batch_sms(batch, device)
            # sms: [B, T, 102], z_flat: [B, T, 128], mask: [B, T]

            # Forward: G(sms) → z_pred
            # Separate direction (cosine) and magnitude (log-norm matching) losses
            z_pred = model.forward_G(sms)                          # [B, T, 128]
            cos_fwd = F.cosine_similarity(z_pred, z_flat, dim=-1)  # [B, T]
            loss_fwd_dir = ((1.0 - cos_fwd) * mask).sum() / mask.sum()
            # Magnitude: log-scale norm matching (relative, not absolute)
            pred_norm = z_pred.norm(dim=-1).clamp(min=1e-6)        # [B, T]
            target_norm = z_flat.norm(dim=-1).clamp(min=1e-6)      # [B, T]
            loss_fwd_mag = ((torch.log(pred_norm) - torch.log(target_norm)).pow(2) * mask).sum() / mask.sum()
            loss_fwd = loss_fwd_dir + 0.5 * loss_fwd_mag

            # Reverse: F(z) → sms_pred
            sms_pred = model.forward_F(z_flat)                     # [B, T, 102]
            loss_rev = ((sms_pred - sms).pow(2).mean(dim=-1) * mask).sum() / mask.sum()

            # Cycle forward: G(F(z)) → z
            z_cycle = model.forward_G(sms_pred)                    # [B, T, 128]
            cos_cyc = F.cosine_similarity(z_cycle, z_flat, dim=-1) # [B, T]
            loss_cyc_dir = ((1.0 - cos_cyc) * mask).sum() / mask.sum()
            cyc_norm = z_cycle.norm(dim=-1).clamp(min=1e-6)
            loss_cyc_mag = ((torch.log(cyc_norm) - torch.log(target_norm)).pow(2) * mask).sum() / mask.sum()
            loss_cyc_fwd = loss_cyc_dir + 0.5 * loss_cyc_mag

            # Cycle reverse: F(G(sms)) → sms
            sms_cycle = model.forward_F(z_pred)                    # [B, T, 102]
            loss_cyc_rev = ((sms_cycle - sms).pow(2).mean(dim=-1) * mask).sum() / mask.sum()

            # Warm up cycle loss
            cycle_w = min(1.0, epoch / 50.0) * cycle_weight
            loss = loss_fwd + loss_rev + cycle_w * (loss_cyc_fwd + loss_cyc_rev)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_fwd += loss_fwd.item() * B
            total_rev += loss_rev.item() * B
            total_cyc_fwd += loss_cyc_fwd.item() * B
            total_cyc_rev += loss_cyc_rev.item() * B
            total_samples += B

            pbar.set_postfix_str(
                f"fwd={total_fwd/total_samples:.4f} "
                f"rev={total_rev/total_samples:.4f}")

        pbar.close()
        scheduler.step()

        print(f"  Epoch {epoch:3d}: "
              f"fwd={total_fwd/total_samples:.6f}  "
              f"rev={total_rev/total_samples:.6f}  "
              f"cyc_f={total_cyc_fwd/total_samples:.6f}  "
              f"cyc_r={total_cyc_rev/total_samples:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    return model


# ============================================================
# Testing
# ============================================================

def test_bidirectional(model, dcae, test_data, output_dir, device='cuda'):
    """Test all four directions."""
    print("\n" + "=" * 60)
    print("TESTING BIDIRECTIONAL CODEC")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    fwd_results = []
    rev_results = []
    cyc_fwd_results = []
    cyc_rev_results = []

    for i, sample in enumerate(test_data[:10]):
        sms = sample['sms_norm'].unsqueeze(0).to(device)
        z = sample['z_sms'].unsqueeze(0).to(device)
        z_flat = model.z_to_flat(z)

        with torch.no_grad():
            # Forward
            z_pred = model.forward_G(sms)
            fwd_cos = F.cosine_similarity(z_pred.flatten().unsqueeze(0),
                                           z_flat.flatten().unsqueeze(0)).item()
            fwd_mse = F.mse_loss(z_pred, z_flat).item()
            # Magnitude ratio: how well G matches the target's norm
            mag_ratio = (z_pred.norm() / z_flat.norm().clamp(min=1e-6)).item()

            # Reverse
            sms_pred = model.forward_F(z_flat)
            rev_mse = F.mse_loss(sms_pred, sms).item()

            # Cycle forward: z → sms → z
            z_cycle = model.forward_G(sms_pred)
            cyc_fwd_cos = F.cosine_similarity(z_cycle.flatten().unsqueeze(0),
                                               z_flat.flatten().unsqueeze(0)).item()

            # Cycle reverse: sms → z → sms
            sms_cycle = model.forward_F(z_pred)
            cyc_rev_mse = F.mse_loss(sms_cycle, sms).item()

        print(f"  Sample {i}: fwd_cos={fwd_cos:.4f}  mag={mag_ratio:.3f}  rev_mse={rev_mse:.6f}  "
              f"cyc_z_cos={cyc_fwd_cos:.4f}  cyc_sms_mse={cyc_rev_mse:.6f}")

        fwd_results.append(fwd_cos)
        rev_results.append(rev_mse)
        cyc_fwd_results.append(cyc_fwd_cos)
        cyc_rev_results.append(cyc_rev_mse)

        # Decode z_pred and z_gt to audio for listening
        z_pred_4d = model.flat_to_z(z_pred)
        z_gt_4d = z
        n_audio = sample.get('n_audio_samples', None)

        pred_audio = decode_latent(dcae, z_pred_4d.squeeze(0), device, n_audio_samples=n_audio)
        gt_audio = decode_latent(dcae, z_gt_4d.squeeze(0), device, n_audio_samples=n_audio)

        torchaudio.save(str(output_dir / f"sample{i}_fwd_pred.wav"),
                       torch.from_numpy(pred_audio).unsqueeze(0), SAMPLE_RATE)
        torchaudio.save(str(output_dir / f"sample{i}_fwd_gt.wav"),
                       torch.from_numpy(gt_audio).unsqueeze(0), SAMPLE_RATE)

    print(f"\n  Forward avg cos_sim:    {np.mean(fwd_results):.4f}")
    print(f"  Reverse avg MSE:        {np.mean(rev_results):.6f}")
    print(f"  Cycle z avg cos_sim:    {np.mean(cyc_fwd_results):.4f}")
    print(f"  Cycle SMS avg MSE:      {np.mean(cyc_rev_results):.6f}")

    return {
        'fwd_cos': np.mean(fwd_results),
        'rev_mse': np.mean(rev_results),
        'cyc_fwd_cos': np.mean(cyc_fwd_results),
        'cyc_rev_mse': np.mean(cyc_rev_results),
    }


def decode_latent(dcae, z, device='cuda', n_audio_samples=None):
    """Decode z [8, 16, T] → audio numpy array."""
    z_4d = z.unsqueeze(0) if z.dim() == 3 else z
    T = z_4d.shape[-1]
    # Use provided audio length, or estimate from DCAE frame rate (~10.8 fps)
    if n_audio_samples is None:
        n_audio_samples = int(T * SAMPLE_RATE / 10.8)
    audio_lengths = torch.tensor([n_audio_samples], device=device)
    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return wavs[0].mean(dim=0).cpu().numpy()


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda'

    print("=" * 60)
    print("PHASE 1: BIDIRECTIONAL SMS-Z CODEC")
    print("=" * 60)
    print()
    print("Goal: Perfect invertible mapping between SMS params and DCAE z")
    print("Key:  z_sms = DCAE_encode(additive_synth(SMS)) — zero information gap")
    print(f"SMS dim: {SMS_DIM}  |  Z dim: 128  |  Compression: {'ON' if USE_COMPRESSION else 'OFF (raw 128 sines)'}")

    output_dir = Path("/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/test_outputs/bidirectional_sms_z")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = "/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json"
    cache_path = output_dir / "data_cache.pt"

    if cache_path.exists():
        print(f"\nLoading cached data from {cache_path}...")
        data = torch.load(cache_path, weights_only=False, map_location='cpu')
        print(f"  Loaded {len(data)} cached samples")

        # Load DCAE for testing only
        print("\nLoading DCAE for testing...")
        dcae = MusicDCAE(
            dcae_checkpoint_path=DCAE_PATH,
            vocoder_checkpoint_path=VOCODER_PATH,
        ).to(device)
        dcae.eval()
        print("DCAE loaded!")
    else:
        # ---- Phase 1a: CPU parallel - render SMS to audio ----
        # Must happen BEFORE loading DCAE to avoid CUDA fork deadlock
        cpu_results = prepare_cpu(manifest_path, max_samples=2000)

        # ---- Phase 1b: Load DCAE, encode rendered audio → z_sms ----
        print("\nLoading DCAE for encoding...")
        dcae = MusicDCAE(
            dcae_checkpoint_path=DCAE_PATH,
            vocoder_checkpoint_path=VOCODER_PATH,
        ).to(device)
        dcae.eval()
        print("DCAE loaded!")

        data = prepare_gpu(cpu_results, dcae, device)
        del cpu_results  # free memory

        # Cache for next run
        print(f"\nCaching {len(data)} samples to {cache_path}...")
        torch.save(data, cache_path)
        print("  Cached!")

    # Split
    n_test = min(50, len(data) // 10)
    train_data = data[:-n_test]
    test_data = data[-n_test:]
    print(f"\nTrain: {len(train_data)}, Test: {len(test_data)}")

    # Free DCAE during training
    dcae = dcae.cpu()
    torch.cuda.empty_cache()
    print("DCAE moved to CPU (freeing GPU for training)")

    # ---- Phase 1b: Train bidirectional codec ----
    model = train_bidirectional(train_data, device, epochs=200, batch_size=32)

    torch.save(model.state_dict(), output_dir / "bidirectional_codec.pt")
    print(f"\nModel saved to {output_dir / 'bidirectional_codec.pt'}")

    # ---- Phase 1c: Test ----
    print("\nReloading DCAE for testing...")
    dcae = dcae.to(device)
    dcae.eval()

    results = test_bidirectional(model, dcae, test_data, output_dir, device)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS")
    print("=" * 60)
    print(f"\n  Forward (SMS→z):     cos_sim = {results['fwd_cos']:.4f}  {'PASS' if results['fwd_cos'] > 0.99 else 'NEEDS WORK'}")
    print(f"  Reverse (z→SMS):     MSE = {results['rev_mse']:.6f}  {'PASS' if results['rev_mse'] < 0.001 else 'NEEDS WORK'}")
    print(f"  Cycle (z→SMS→z):     cos_sim = {results['cyc_fwd_cos']:.4f}  {'PASS' if results['cyc_fwd_cos'] > 0.99 else 'NEEDS WORK'}")
    print(f"  Cycle (SMS→z→SMS):   MSE = {results['cyc_rev_mse']:.6f}  {'PASS' if results['cyc_rev_mse'] < 0.001 else 'NEEDS WORK'}")

    print(f"\nOutputs: {output_dir}")
    print("  sample*_fwd_pred.wav = G(sms) decoded")
    print("  sample*_fwd_gt.wav   = z_sms decoded (ground truth)")

    if results['fwd_cos'] > 0.99 and results['cyc_fwd_cos'] > 0.99:
        print("\n  Phase 1 PASSED — ready for Phase 2 (extend operations)")
    else:
        print("\n  Phase 1 needs more training or architecture changes")


if __name__ == "__main__":
    main()
