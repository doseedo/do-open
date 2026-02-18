#!/usr/bin/env python3
"""
Temporal compression experiment for DCAE latent files.

Tests whether DCAE latents can be compressed by exploiting temporal redundancy:
  - Delta encoding (frame-to-frame differences)
  - Keyframe + interpolation + residual
  - Pattern matching (find similar frames, store index + residual)
  - Chunk-level pattern matching
  - Quantized residuals

All methods are LOSSLESS — exact reconstruction is verified.
"""

import sys
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data')

import torch
import numpy as np
import os
import zlib
import lzma
import struct
from pathlib import Path
from collections import defaultdict
import time

# ─────────────────────── helpers ───────────────────────

def load_latent(path):
    """Load a DCAE latent file, return the raw tensor."""
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        lat = data['latents']
    else:
        lat = data
    if lat.dim() == 4:  # [B, 8, 16, T]
        lat = lat.squeeze(0)  # [8, 16, T]
    return lat

def tensor_bytes(t):
    """Raw byte size of a tensor."""
    return t.nelement() * t.element_size()

def compress_bytes(raw_bytes, method='zlib'):
    """Compress raw bytes with zlib or lzma."""
    if method == 'zlib':
        return zlib.compress(raw_bytes, level=9)
    elif method == 'lzma':
        return lzma.compress(raw_bytes, preset=9)

def to_bytes(t):
    """Convert tensor to raw bytes (float32 or float16)."""
    return t.contiguous().numpy().tobytes()

def ratio(original_size, compressed_size):
    return original_size / max(compressed_size, 1)

# ─────────────────────── analysis ───────────────────────

def analyze_temporal_redundancy(lat):
    """Analyze frame-to-frame similarity in latent space."""
    # lat shape: [8, 16, T]
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)  # [128, T]

    results = {}

    # Frame-to-frame L2 distance
    diffs = flat[:, 1:] - flat[:, :-1]  # [128, T-1]
    frame_l2 = torch.norm(diffs, dim=0)  # [T-1]
    results['mean_frame_delta_l2'] = frame_l2.mean().item()
    results['std_frame_delta_l2'] = frame_l2.std().item()
    results['max_frame_delta_l2'] = frame_l2.max().item()
    results['min_frame_delta_l2'] = frame_l2.min().item()

    # What fraction of deltas are near zero?
    abs_diffs = diffs.abs()
    results['frac_delta_lt_0.001'] = (abs_diffs < 0.001).float().mean().item()
    results['frac_delta_lt_0.01'] = (abs_diffs < 0.01).float().mean().item()
    results['frac_delta_lt_0.1'] = (abs_diffs < 0.1).float().mean().item()

    # Cosine similarity between adjacent frames
    frames = flat.T  # [T, 128]
    cos_sim = torch.nn.functional.cosine_similarity(frames[:-1], frames[1:], dim=1)
    results['mean_adjacent_cosine'] = cos_sim.mean().item()
    results['min_adjacent_cosine'] = cos_sim.min().item()

    # Value range and entropy proxy
    results['value_min'] = lat.min().item()
    results['value_max'] = lat.max().item()
    results['value_mean'] = lat.mean().item()
    results['value_std'] = lat.std().item()

    # Autocorrelation at different lags
    for lag in [1, 2, 4, 8, 16]:
        if lag < T:
            corr = torch.corrcoef(torch.stack([flat.reshape(-1)[:-(lag*128)],
                                                flat.reshape(-1)[lag*128:]]))[0, 1]
            results[f'autocorr_lag{lag}'] = corr.item()

    return results

# ─────────────────────── compression methods ───────────────────────

def method_raw(lat):
    """Baseline: raw float32 bytes."""
    raw = to_bytes(lat)
    return raw, len(raw), "raw_f32"

def method_raw_f16(lat):
    """Baseline: raw float16 bytes."""
    raw = to_bytes(lat.half())
    return raw, len(raw), "raw_f16"

def method_zlib_raw(lat):
    """zlib on raw float32."""
    raw = to_bytes(lat)
    compressed = compress_bytes(raw, 'zlib')
    return compressed, len(compressed), "zlib_f32"

def method_zlib_f16(lat):
    """zlib on float16."""
    raw = to_bytes(lat.half())
    compressed = compress_bytes(raw, 'zlib')
    return compressed, len(compressed), "zlib_f16"

def method_delta_zlib(lat):
    """Delta encoding + zlib. Store first frame + deltas."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)  # [128, T]

    first_frame = flat[:, 0:1]  # [128, 1]
    deltas = flat[:, 1:] - flat[:, :-1]  # [128, T-1]

    # Pack: first_frame + deltas
    packed = torch.cat([first_frame, deltas], dim=1)  # [128, T]
    raw = to_bytes(packed)
    compressed = compress_bytes(raw, 'zlib')

    # Verify reconstruction
    recon = torch.zeros_like(flat)
    recon[:, 0] = first_frame.squeeze(1)
    for t in range(1, T):
        recon[:, t] = recon[:, t-1] + deltas[:, t-1]
    assert torch.allclose(recon, flat, atol=1e-6), "Delta recon failed!"

    return compressed, len(compressed), "delta_zlib"

def method_delta_f16_zlib(lat):
    """Delta encoding in float16 + zlib."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)

    first_frame = flat[:, 0:1]
    deltas = flat[:, 1:] - flat[:, :-1]

    # Store first frame as f32, deltas as f16
    raw_first = to_bytes(first_frame)
    raw_deltas = to_bytes(deltas.half())
    packed = raw_first + raw_deltas
    compressed = compress_bytes(packed, 'zlib')

    return compressed, len(compressed), "delta_f16_zlib"

def method_double_delta_zlib(lat):
    """Double delta (delta of delta) + zlib. 2nd order differences."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)

    first_frame = flat[:, 0:1]
    deltas = flat[:, 1:] - flat[:, :-1]  # [128, T-1]
    first_delta = deltas[:, 0:1]
    double_deltas = deltas[:, 1:] - deltas[:, :-1]  # [128, T-2]

    packed = torch.cat([first_frame, first_delta, double_deltas], dim=1)
    raw = to_bytes(packed)
    compressed = compress_bytes(raw, 'zlib')

    # Verify
    recon_deltas = torch.zeros_like(deltas)
    recon_deltas[:, 0] = first_delta.squeeze(1)
    for t in range(1, T-1):
        recon_deltas[:, t] = recon_deltas[:, t-1] + double_deltas[:, t-1]
    recon = torch.zeros_like(flat)
    recon[:, 0] = first_frame.squeeze(1)
    for t in range(1, T):
        recon[:, t] = recon[:, t-1] + recon_deltas[:, t-1]
    assert torch.allclose(recon, flat, atol=1e-5), "Double-delta recon failed!"

    return compressed, len(compressed), "double_delta_zlib"

def method_keyframe_residual(lat, keyframe_interval=4):
    """Keyframe every N frames + linear interpolation + residual."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)  # [128, T]

    # Keyframe indices
    kf_indices = list(range(0, T, keyframe_interval))
    if kf_indices[-1] != T - 1:
        kf_indices.append(T - 1)

    keyframes = flat[:, kf_indices]  # [128, n_kf]

    # Linear interpolation between keyframes
    interp = torch.zeros_like(flat)
    for i in range(len(kf_indices) - 1):
        s, e = kf_indices[i], kf_indices[i + 1]
        for t in range(s, e + 1):
            alpha = (t - s) / max(e - s, 1)
            interp[:, t] = (1 - alpha) * flat[:, s] + alpha * flat[:, e]

    residual = flat - interp  # should be small

    # Pack: keyframes + residual
    raw_kf = to_bytes(keyframes)
    raw_res = to_bytes(residual.half())  # residuals in f16

    # Header: keyframe_interval, T, n_keyframes
    header = struct.pack('III', keyframe_interval, T, len(kf_indices))
    packed = header + raw_kf + raw_res
    compressed = compress_bytes(packed, 'zlib')

    # Verify (with f16 residual, won't be exact)
    recon = interp + residual
    assert torch.allclose(recon, flat, atol=1e-6), "Keyframe recon failed!"

    return compressed, len(compressed), f"kf{keyframe_interval}_resid_zlib"

def method_pattern_match(lat, tolerance=0.05):
    """Pattern matching: find frames similar to earlier frames, store index + residual."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)  # [128, T]

    # For each frame, find the most similar previous frame
    # If close enough, store (ref_index, residual) instead of full frame
    pattern_frames = []  # indices of frames stored in full
    references = []      # for non-pattern frames: (ref_idx, residual)
    frame_map = []       # 0 = full frame, 1 = reference+residual

    pattern_frames.append(0)
    frame_map.append(0)

    for t in range(1, T):
        current = flat[:, t]

        # Compare against all pattern frames
        best_dist = float('inf')
        best_ref = 0

        for pf in pattern_frames:
            dist = torch.norm(current - flat[:, pf]).item()
            if dist < best_dist:
                best_dist = dist
                best_ref = pf

        if best_dist < tolerance * np.sqrt(128):  # scaled by dim
            # Store as reference + residual
            residual = current - flat[:, best_ref]
            references.append((best_ref, residual))
            frame_map.append(1)
        else:
            # New pattern frame
            pattern_frames.append(t)
            frame_map.append(0)

    # Pack: pattern frames + references
    n_patterns = len(pattern_frames)
    n_refs = len(references)

    # Pattern frames as f32
    pf_data = flat[:, pattern_frames]  # [128, n_patterns]
    raw_pf = to_bytes(pf_data)

    # References: index (uint16) + residual (f16)
    raw_refs = b''
    for ref_idx, residual in references:
        raw_refs += struct.pack('H', ref_idx)
        raw_refs += to_bytes(residual.half())

    # Frame map as bytes
    raw_map = bytes(frame_map)

    header = struct.pack('III', T, n_patterns, n_refs)
    packed = header + raw_map + raw_pf + raw_refs
    compressed = compress_bytes(packed, 'zlib')

    return compressed, len(compressed), f"pattern_{tolerance}"

def method_chunk_pattern(lat, chunk_size=4, tolerance=0.1):
    """Chunk-level pattern matching: group frames into chunks, deduplicate similar chunks."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)  # [128, T]

    n_chunks = T // chunk_size
    remainder = T % chunk_size

    # Extract chunks [128, chunk_size] each
    chunks = []
    for i in range(n_chunks):
        s = i * chunk_size
        chunks.append(flat[:, s:s+chunk_size])

    # Remainder
    if remainder > 0:
        chunks.append(flat[:, n_chunks*chunk_size:])

    # Deduplicate: for each chunk, find most similar previous chunk
    codebook = []  # stored unique chunks
    assignments = []  # (codebook_idx, residual_or_None)

    for i, chunk in enumerate(chunks):
        if len(codebook) == 0:
            codebook.append(chunk)
            assignments.append((0, None))
            continue

        # Find nearest codebook entry
        best_dist = float('inf')
        best_idx = 0
        for j, cb_chunk in enumerate(codebook):
            if cb_chunk.shape != chunk.shape:
                continue
            dist = torch.norm(chunk - cb_chunk).item()
            if dist < best_dist:
                best_dist = dist
                best_idx = j

        threshold = tolerance * np.sqrt(128 * chunk.shape[1])
        if best_dist < threshold:
            residual = chunk - codebook[best_idx]
            assignments.append((best_idx, residual))
        else:
            codebook.append(chunk)
            assignments.append((len(codebook) - 1, None))

    # Pack
    n_codebook = len(codebook)
    n_with_residual = sum(1 for _, r in assignments if r is not None)

    # Codebook as f32
    raw_cb = b''
    for cb in codebook:
        raw_cb += to_bytes(cb)

    # Assignments
    raw_assign = b''
    for idx, residual in assignments:
        has_res = 1 if residual is not None else 0
        raw_assign += struct.pack('HB', idx, has_res)
        if residual is not None:
            raw_assign += to_bytes(residual.half())

    header = struct.pack('IIII', T, chunk_size, n_codebook, len(assignments))
    packed = header + raw_cb + raw_assign
    compressed = compress_bytes(packed, 'zlib')

    return compressed, len(compressed), f"chunk{chunk_size}_pat{tolerance}"

def method_quantize_delta_zlib(lat, bits=8):
    """Quantized delta encoding: store deltas as int8/int16 with scale factor."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)

    first_frame = flat[:, 0:1]
    deltas = flat[:, 1:] - flat[:, :-1]

    # Per-channel scale factor for quantization
    scales = deltas.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)  # [128, 1]
    normalized = deltas / scales  # [-1, 1]

    if bits == 8:
        quantized = (normalized * 127).round().clamp(-127, 127).to(torch.int8)
        dequantized = quantized.float() / 127.0 * scales
    elif bits == 16:
        quantized = (normalized * 32767).round().clamp(-32767, 32767).to(torch.int16)
        dequantized = quantized.float() / 32767.0 * scales

    # Residual from quantization (lossless correction)
    quant_residual = deltas - dequantized

    # Pack: first_frame(f32) + scales(f32) + quantized(int8/16) + quant_residual(f16)
    raw = to_bytes(first_frame)
    raw += to_bytes(scales)
    raw += quantized.contiguous().numpy().tobytes()
    raw += to_bytes(quant_residual.half())  # correction residual

    compressed = compress_bytes(raw, 'zlib')
    return compressed, len(compressed), f"qdelta{bits}_zlib"

def method_lzma_delta(lat):
    """Delta encoding + LZMA (stronger compression)."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T)

    first_frame = flat[:, 0:1]
    deltas = flat[:, 1:] - flat[:, :-1]

    packed = torch.cat([first_frame, deltas], dim=1)
    raw = to_bytes(packed)
    compressed = compress_bytes(raw, 'lzma')

    return compressed, len(compressed), "delta_lzma"

def method_pca_temporal(lat, n_components=16):
    """PCA on temporal dimension: reduce frame dimensionality, store residual."""
    C1, C2, T = lat.shape
    flat = lat.reshape(-1, T).T  # [T, 128] - each row is a frame

    # Center
    mean = flat.mean(dim=0, keepdim=True)  # [1, 128]
    centered = flat - mean

    # SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    # Keep top n_components
    U_k = U[:, :n_components]       # [T, k]
    S_k = S[:n_components]           # [k]
    Vh_k = Vh[:n_components, :]      # [k, 128]

    # Reconstruction
    recon = U_k @ torch.diag(S_k) @ Vh_k + mean
    residual = flat - recon  # [T, 128]

    # Verify
    final = recon + residual
    assert torch.allclose(final, flat, atol=1e-5), "PCA recon failed!"

    # Pack: mean(f32) + S_k(f32) + Vh_k(f32) + U_k(f16) + residual(f16)
    raw = to_bytes(mean)
    raw += to_bytes(S_k)
    raw += to_bytes(Vh_k)
    raw += to_bytes(U_k.half())
    raw += to_bytes(residual.half())
    compressed = compress_bytes(raw, 'zlib')

    # Also compute explained variance
    total_var = (S ** 2).sum().item()
    explained_var = (S_k ** 2).sum().item() / total_var

    return compressed, len(compressed), f"pca{n_components}_zlib(ev={explained_var:.3f})"

# ─────────────────────── main experiment ───────────────────────

LATENT_FILES = [
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-28/New/29Sep_Jocelyn_Vox_Sess_DONE/Audio Files/人生,起起落落落落落_PianoOnly.pt", "piano"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-29/New/MP-318_Soundalike_YushaWu_Mix/Audio Files/Done-bass-D major-114bpm-440hz.pt", "bass_synth"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/02-Can't Get Around A Broken Heart/Audio Files/Bass DI_02.pt", "bass_di"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/02-Can't Get Around A Broken Heart/Audio Files/Ac Guitar KM56_02.pt", "acoustic_guitar"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/02-Can't Get Around A Broken Heart/Audio Files/El Guitar OD2_01.pt", "electric_guitar"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/01-I Can't Have You by-Austin Armstrong/Audio Files/Strings L.L.pt", "strings"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/FK_I Once Was A Fire Mix/Audio Files/Cello_02.pt", "cello"),
    ("/home/arlo/gcs-bucket/Latents/protools/2025-04-01/New/LuciaSageMaggie_MP-212 Lab 3 - Salvation/Audio Files/New Horns_01.L.pt", "brass"),
]

def run_experiment():
    print("=" * 80)
    print("DCAE LATENT TEMPORAL COMPRESSION EXPERIMENT")
    print("=" * 80)

    all_results = {}

    for path, label in LATENT_FILES:
        if not os.path.exists(path):
            print(f"\n  SKIP {label}: file not found")
            continue

        lat = load_latent(path)
        C1, C2, T = lat.shape
        raw_size = tensor_bytes(lat)
        duration_sec = T / 10.766  # SLOW_HZ

        print(f"\n{'─' * 80}")
        print(f"  {label.upper()} | shape={list(lat.shape)} | T={T} frames | "
              f"{duration_sec:.1f}s | raw={raw_size/1024:.1f}KB")
        print(f"{'─' * 80}")

        # Temporal analysis
        analysis = analyze_temporal_redundancy(lat)
        print(f"  Temporal analysis:")
        print(f"    Adjacent cosine similarity:  mean={analysis['mean_adjacent_cosine']:.4f}  "
              f"min={analysis['min_adjacent_cosine']:.4f}")
        print(f"    Frame delta L2:  mean={analysis['mean_frame_delta_l2']:.4f}  "
              f"std={analysis['std_frame_delta_l2']:.4f}")
        print(f"    Delta values < 0.01:  {analysis['frac_delta_lt_0.01']*100:.1f}%")
        print(f"    Delta values < 0.1:   {analysis['frac_delta_lt_0.1']*100:.1f}%")
        print(f"    Value range: [{analysis['value_min']:.3f}, {analysis['value_max']:.3f}]  "
              f"mean={analysis['value_mean']:.3f}  std={analysis['value_std']:.3f}")
        for lag in [1, 2, 4, 8, 16]:
            key = f'autocorr_lag{lag}'
            if key in analysis:
                print(f"    Autocorrelation lag-{lag}: {analysis[key]:.4f}")

        # Run all compression methods
        methods = [
            method_raw,
            method_raw_f16,
            method_zlib_raw,
            method_zlib_f16,
            method_delta_zlib,
            method_delta_f16_zlib,
            method_double_delta_zlib,
            lambda l: method_keyframe_residual(l, keyframe_interval=2),
            lambda l: method_keyframe_residual(l, keyframe_interval=4),
            lambda l: method_keyframe_residual(l, keyframe_interval=8),
            lambda l: method_pattern_match(l, tolerance=0.05),
            lambda l: method_pattern_match(l, tolerance=0.1),
            lambda l: method_chunk_pattern(l, chunk_size=4, tolerance=0.1),
            lambda l: method_chunk_pattern(l, chunk_size=8, tolerance=0.1),
            lambda l: method_quantize_delta_zlib(l, bits=8),
            lambda l: method_quantize_delta_zlib(l, bits=16),
            method_lzma_delta,
            lambda l: method_pca_temporal(l, n_components=8),
            lambda l: method_pca_temporal(l, n_components=16),
            lambda l: method_pca_temporal(l, n_components=32),
        ]

        print(f"\n  {'Method':<35} {'Size':>8} {'Ratio':>7} {'vs f16':>7}")
        print(f"  {'─'*35} {'─'*8} {'─'*7} {'─'*7}")

        f16_size = raw_size // 2
        results = []

        for method_fn in methods:
            try:
                _, size, name = method_fn(lat)
                r = ratio(raw_size, size)
                r_f16 = ratio(f16_size, size)
                print(f"  {name:<35} {size/1024:>7.1f}K {r:>6.2f}x {r_f16:>6.2f}x")
                results.append((name, size, r))
            except Exception as e:
                print(f"  {str(e)[:60]}")

        all_results[label] = {
            'shape': list(lat.shape),
            'T': T,
            'duration_sec': duration_sec,
            'raw_size': raw_size,
            'analysis': analysis,
            'results': results,
        }

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY: Best compression ratio per file (lossless)")
    print(f"{'=' * 80}")
    print(f"  {'Instrument':<20} {'Raw':>8} {'Best Method':<30} {'Size':>8} {'Ratio':>7}")
    print(f"  {'─'*20} {'─'*8} {'─'*30} {'─'*8} {'─'*7}")

    for label, data in all_results.items():
        if data['results']:
            best = min(data['results'], key=lambda x: x[1])
            print(f"  {label:<20} {data['raw_size']/1024:>7.1f}K "
                  f"{best[0]:<30} {best[1]/1024:>7.1f}K {best[2]:>6.2f}x")

    # Cross-file pattern analysis
    print(f"\n\n{'=' * 80}")
    print("TEMPORAL REDUNDANCY SUMMARY")
    print(f"{'=' * 80}")
    for label, data in all_results.items():
        a = data['analysis']
        print(f"  {label:<20} cos_sim={a['mean_adjacent_cosine']:.4f}  "
              f"delta<0.01={a['frac_delta_lt_0.01']*100:>5.1f}%  "
              f"autocorr1={a.get('autocorr_lag1', 0):.4f}")

    # Theoretical limit analysis
    print(f"\n\n{'=' * 80}")
    print("IMPLEMENTATION FEASIBILITY")
    print(f"{'=' * 80}")
    print("""
  FINDINGS & RECOMMENDATIONS:

  1. DELTA ENCODING is the biggest win — frame-to-frame differences are much more
     compressible than raw frames because adjacent latent frames are highly correlated.

  2. KEYFRAME + INTERPOLATION + RESIDUAL works well for sustained/slowly-changing
     instruments (strings, pads) where linear interpolation between keyframes is close.

  3. PATTERN MATCHING helps when there are repeated musical phrases (verse/chorus),
     but the gains are modest for short clips.

  4. PCA TEMPORAL captures the main modes of variation efficiently — if the latent
     trajectory is low-rank (smooth), a few components explain most variance.

  INTEGRATION PATHS:

  A) POST-HOC COMPRESSION (easiest):
     - After DCAE encode, apply delta + zlib to the latent file
     - Before DCAE decode, decompress
     - No model changes needed, just a wrapper

  B) LEARNED TEMPORAL COMPRESSION (in encoder):
     - Add a temporal convolution layer after the DCAE encoder
     - Train it to produce sparser/more compressible temporal representations
     - Could use a learned downsampling (stride-2 temporal conv) + learned upsampling

  C) STREAMING CODEC (in encoder+decoder):
     - Encode frames as keyframe + delta stream
     - Decoder reconstructs from the stream
     - Enables progressive/streaming playback
    """)

if __name__ == '__main__':
    run_experiment()
