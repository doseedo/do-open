#!/usr/bin/env python3
# /home/arlo/Data/clarifier/align_pairs.py
# Apache 2.0
# DTW alignment for synthetic-real latent pairs

"""
Since performer model timing may not exactly match real recordings,
this script aligns the real latent to match synthetic timing using DTW.

Usage:
    python align_pairs.py \
        --input_dir /path/to/clarifier_pairs \
        --output_dir /path/to/clarifier_pairs_aligned \
        --method dtw

Methods:
    - dtw: Dynamic Time Warping on amplitude envelopes
    - soft_dtw: Differentiable Soft-DTW (smoother)
    - xcorr: Cross-correlation alignment (simple shift)
    - none: No alignment, just copy (for ablation)
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

# Optional DTW libraries
try:
    from dtaidistance import dtw
    HAVE_DTAIDISTANCE = True
except ImportError:
    HAVE_DTAIDISTANCE = False
    print("Warning: dtaidistance not installed. Using fallback DTW.")

try:
    from soft_dtw_cuda import SoftDTW
    HAVE_SOFT_DTW = True
except ImportError:
    HAVE_SOFT_DTW = False


def get_amplitude_envelope(latent: torch.Tensor) -> torch.Tensor:
    """
    Extract amplitude proxy from latent for alignment.

    Args:
        latent: [8, 16, T] or [B, 8, 16, T]

    Returns:
        amp: [T] amplitude envelope
    """
    if latent.dim() == 4:
        latent = latent.squeeze(0)

    # Use L2 norm across channels and height as amplitude proxy
    amp = latent.pow(2).sum(dim=(0, 1)).sqrt()  # [T]
    return amp


def simple_dtw(seq1: np.ndarray, seq2: np.ndarray) -> list:
    """
    Simple DTW implementation (fallback when dtaidistance not available).

    Returns list of (i, j) index pairs for alignment path.
    """
    n, m = len(seq1), len(seq2)

    # Cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [
                (dtw_matrix[i - 1, j], i - 1, j),
                (dtw_matrix[i, j - 1], i, j - 1),
                (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
            ]
            _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()
    return path


def dtw_align_latents(
    synthetic: torch.Tensor,
    real: torch.Tensor,
    window_constraint: Optional[int] = None,
) -> torch.Tensor:
    """
    Align real latent to synthetic using DTW on amplitude envelopes.

    Args:
        synthetic: [8, 16, T_syn]
        real: [8, 16, T_real]
        window_constraint: Sakoe-Chiba band width (None = no constraint)

    Returns:
        aligned_real: [8, 16, T_syn] real latent warped to match synthetic timing
    """
    # Get amplitude envelopes
    syn_amp = get_amplitude_envelope(synthetic).numpy()
    real_amp = get_amplitude_envelope(real).numpy()

    T_syn = len(syn_amp)
    T_real = len(real_amp)

    # Normalize
    syn_amp = (syn_amp - syn_amp.mean()) / (syn_amp.std() + 1e-8)
    real_amp = (real_amp - real_amp.mean()) / (real_amp.std() + 1e-8)

    # Get DTW path
    if HAVE_DTAIDISTANCE:
        if window_constraint:
            path = dtw.warping_path(syn_amp, real_amp, window=window_constraint)
        else:
            path = dtw.warping_path(syn_amp, real_amp)
    else:
        path = simple_dtw(syn_amp, real_amp)

    # Create aligned real latent
    aligned_real = torch.zeros_like(synthetic)
    counts = torch.zeros(T_syn)

    for syn_idx, real_idx in path:
        if 0 <= syn_idx < T_syn and 0 <= real_idx < T_real:
            aligned_real[..., syn_idx] += real[..., real_idx]
            counts[syn_idx] += 1

    # Average where multiple real frames map to same synthetic frame
    counts = counts.clamp(min=1)
    aligned_real = aligned_real / counts.view(1, 1, -1)

    return aligned_real


def xcorr_align_latents(
    synthetic: torch.Tensor,
    real: torch.Tensor,
    max_shift: int = 50,
) -> torch.Tensor:
    """
    Align real latent to synthetic using cross-correlation (simple global shift).

    Args:
        synthetic: [8, 16, T_syn]
        real: [8, 16, T_real]
        max_shift: Maximum shift in frames

    Returns:
        aligned_real: [8, 16, T_syn]
    """
    syn_amp = get_amplitude_envelope(synthetic)
    real_amp = get_amplitude_envelope(real)

    T_syn = syn_amp.shape[0]
    T_real = real_amp.shape[0]

    # Pad for correlation
    padded_real_amp = F.pad(real_amp.view(1, 1, -1), (max_shift, max_shift)).squeeze()

    best_shift = 0
    best_corr = -float('inf')

    for shift in range(-max_shift, max_shift + 1):
        # Get shifted real amplitude
        start = max_shift + shift
        end = start + T_syn
        shifted = padded_real_amp[start:end]

        if len(shifted) != T_syn:
            continue

        # Compute correlation
        corr = (syn_amp * shifted).sum().item()
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    # Apply shift to real latent
    if best_shift > 0:
        # Real is ahead, shift left (pad right)
        aligned_real = F.pad(real, (0, best_shift))[..., best_shift:best_shift + T_syn]
    elif best_shift < 0:
        # Real is behind, shift right (pad left)
        aligned_real = F.pad(real, (-best_shift, 0))[..., :T_syn]
    else:
        aligned_real = real[..., :T_syn]

    # Ensure correct length
    if aligned_real.shape[-1] < T_syn:
        aligned_real = F.pad(aligned_real, (0, T_syn - aligned_real.shape[-1]))
    elif aligned_real.shape[-1] > T_syn:
        aligned_real = aligned_real[..., :T_syn]

    return aligned_real


def align_single_pair(
    pair_path: str,
    output_path: str,
    method: str = "dtw",
    window_constraint: Optional[int] = None,
):
    """
    Align a single pair file and save.
    """
    data = torch.load(pair_path, map_location="cpu", weights_only=False)

    synthetic = data["synthetic"]
    real = data["real"]

    # Handle batch dimension
    if synthetic.dim() == 4:
        synthetic = synthetic.squeeze(0)
    if real.dim() == 4:
        real = real.squeeze(0)

    # Align based on method
    if method == "dtw":
        aligned_real = dtw_align_latents(synthetic, real, window_constraint)
    elif method == "xcorr":
        aligned_real = xcorr_align_latents(synthetic, real)
    elif method == "none":
        # Just crop/pad to match
        T_syn = synthetic.shape[-1]
        if real.shape[-1] < T_syn:
            aligned_real = F.pad(real, (0, T_syn - real.shape[-1]))
        else:
            aligned_real = real[..., :T_syn]
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    # Save aligned pair
    aligned_data = {
        "synthetic": synthetic,
        "real": aligned_real,
        "group_id": data.get("group_id"),
        "subgroup_id": data.get("subgroup_id"),
        "meta": data.get("meta", {}),
        "alignment_method": method,
    }

    torch.save(aligned_data, output_path)


def align_pairs(
    input_dir: str,
    output_dir: str,
    method: str = "dtw",
    window_constraint: Optional[int] = None,
    skip_existing: bool = True,
):
    """
    Align all pairs in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(input_dir)
    pair_files = sorted(input_path.glob("*.pt"))

    print(f"Found {len(pair_files)} pair files to align")
    print(f"Method: {method}, Window constraint: {window_constraint}")

    aligned_count = 0
    skipped_count = 0
    error_count = 0

    for pair_file in tqdm(pair_files):
        output_path = os.path.join(output_dir, pair_file.name)

        if skip_existing and os.path.exists(output_path):
            skipped_count += 1
            continue

        try:
            align_single_pair(
                str(pair_file),
                output_path,
                method=method,
                window_constraint=window_constraint,
            )
            aligned_count += 1
        except Exception as e:
            print(f"Error aligning {pair_file.name}: {e}")
            error_count += 1
            continue

    print(f"Done! Aligned {aligned_count}, skipped {skipped_count}, errors {error_count}")


def visualize_alignment(pair_path: str, output_png: Optional[str] = None):
    """
    Visualize alignment quality for a single pair.
    """
    import matplotlib.pyplot as plt

    data = torch.load(pair_path, map_location="cpu", weights_only=False)

    synthetic = data["synthetic"]
    real = data["real"]

    if synthetic.dim() == 4:
        synthetic = synthetic.squeeze(0)
    if real.dim() == 4:
        real = real.squeeze(0)

    syn_amp = get_amplitude_envelope(synthetic).numpy()
    real_amp = get_amplitude_envelope(real).numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(syn_amp, label="Synthetic", alpha=0.8)
    axes[0].plot(real_amp, label="Real (aligned)", alpha=0.8)
    axes[0].set_title("Amplitude Envelopes After Alignment")
    axes[0].legend()
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Amplitude")

    # Show latent spectrograms
    syn_spec = synthetic.mean(dim=0).numpy()  # [16, T]
    real_spec = real.mean(dim=0).numpy()

    axes[1].imshow(
        np.concatenate([syn_spec, np.zeros((1, syn_spec.shape[1])), real_spec], axis=0),
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )
    axes[1].set_title("Latent Spectrograms (Synthetic | Real)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Height bin")

    plt.tight_layout()

    if output_png:
        plt.savefig(output_png, dpi=150)
        print(f"Saved visualization to {output_png}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Align synthetic-real latent pairs")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory with pair .pt files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for aligned pairs")
    parser.add_argument("--method", type=str, default="dtw",
                        choices=["dtw", "xcorr", "none"],
                        help="Alignment method")
    parser.add_argument("--window_constraint", type=int, default=None,
                        help="Sakoe-Chiba window constraint for DTW")
    parser.add_argument("--no_skip", action="store_true",
                        help="Don't skip existing aligned files")
    parser.add_argument("--visualize", type=str, default=None,
                        help="Visualize a single pair file (path)")

    args = parser.parse_args()

    if args.visualize:
        visualize_alignment(args.visualize)
    else:
        align_pairs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            method=args.method,
            window_constraint=args.window_constraint,
            skip_existing=not args.no_skip,
        )


if __name__ == "__main__":
    main()
