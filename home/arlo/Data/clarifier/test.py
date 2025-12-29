#!/usr/bin/env python3
# /home/arlo/Data/clarifier/test.py
# Apache 2.0
# Inference script for InstrumentClarifier + decode to audio

"""
Usage:
    # Test on a single latent file:
    python test.py \
        --checkpoint /path/to/clarifier_checkpoints/best.pt \
        --input /path/to/synthetic_latent.pt \
        --output_dir /path/to/outputs \
        --group_id 4 --subgroup_id 7 \
        --ace_checkpoint /home/arlo/Data/ACE-Step/checkpoints

    # Test on a directory of latents:
    python test.py \
        --checkpoint /path/to/clarifier_checkpoints/best.pt \
        --input_dir /path/to/synthetic_latents \
        --output_dir /path/to/outputs \
        --ace_checkpoint /home/arlo/Data/ACE-Step/checkpoints

    # Test on pairs (compare synthetic, clarified, real):
    python test.py \
        --checkpoint /path/to/clarifier_checkpoints/best.pt \
        --pairs_dir /path/to/clarifier_pairs_aligned \
        --output_dir /path/to/comparison_outputs \
        --ace_checkpoint /home/arlo/Data/ACE-Step/checkpoints \
        --max_samples 10
"""

import sys
sys.path.insert(0, '/home/arlo/Data/dø')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/clarifier')

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np

from models import InstrumentClarifier, InstrumentClarifierLarge
from dataset import APPROVED_GROUPS, APPROVED_SUBGROUPS

# Import DCAE for audio decoding
try:
    from do.pipeline_do import DoTrainComponents
    HAVE_DCAE = True
except ImportError:
    HAVE_DCAE = False
    print("Warning: DCAE not available. Audio decoding disabled.")


# Grid rates
DCAE_SR, DCAE_HOP = 44100, 4096


def load_clarifier(checkpoint_path: str, model_size: str = "base", device: str = "cuda"):
    """Load trained clarifier model."""
    if model_size == "large":
        model = InstrumentClarifierLarge()
    else:
        model = InstrumentClarifier()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded clarifier from {checkpoint_path}")
    if "model_config" in ckpt:
        print(f"  Input scale: {ckpt['model_config'].get('input_scale', 'N/A')}")
        print(f"  Output scale: {ckpt['model_config'].get('output_scale', 'N/A')}")

    return model


def load_dcae(ace_checkpoint_dir: str, device: str = "cuda"):
    """Load DCAE for decoding latents to audio."""
    if not HAVE_DCAE:
        return None

    comps = DoTrainComponents(checkpoint_dir=ace_checkpoint_dir, dtype="float32")
    comps.device = torch.device(device)
    dcae = comps.load_dcae()
    dcae.to(device).eval()
    print("Loaded DCAE for audio decoding")
    return dcae


@torch.no_grad()
def clarify_latent(
    model: torch.nn.Module,
    latent: torch.Tensor,
    group_id: int,
    subgroup_id: int,
    device: str = "cuda",
    window_size: int = 512,
    overlap: int = 128,
) -> torch.Tensor:
    """
    Apply clarifier to latent with sliding window.

    Args:
        model: Trained InstrumentClarifier
        latent: [8, 16, T] or [B, 8, 16, T] input latent
        group_id: Instrument group index
        subgroup_id: Instrument subgroup index
        window_size: Processing window size
        overlap: Overlap between windows for smooth transitions

    Returns:
        clarified: [8, 16, T] or [B, 8, 16, T] clarified latent
    """
    model.eval()

    # Handle batch dimension
    had_batch = latent.dim() == 4
    if not had_batch:
        latent = latent.unsqueeze(0)

    B, C, H, T = latent.shape
    latent = latent.to(device)

    # Prepare IDs
    g_id = torch.tensor([group_id], device=device, dtype=torch.long).expand(B)
    sg_id = torch.tensor([subgroup_id], device=device, dtype=torch.long).expand(B)

    # If short enough, process in one go
    if T <= window_size:
        clarified = model(latent, g_id, sg_id)
        if not had_batch:
            clarified = clarified.squeeze(0)
        return clarified

    # Sliding window with overlap
    step = window_size - overlap
    clarified = torch.zeros_like(latent)
    weights = torch.zeros(1, 1, 1, T, device=device)

    # Create fade window for blending
    fade = torch.ones(window_size, device=device)
    if overlap > 0:
        fade_in = torch.linspace(0, 1, overlap, device=device)
        fade_out = torch.linspace(1, 0, overlap, device=device)
        fade[:overlap] = fade_in
        fade[-overlap:] = fade_out

    for start in range(0, T, step):
        end = min(start + window_size, T)
        actual_len = end - start

        # Extract window
        window = latent[..., start:end]

        # Pad if needed
        if actual_len < window_size:
            window = F.pad(window, (0, window_size - actual_len))

        # Clarify
        clarified_window = model(window, g_id, sg_id)

        # Remove padding
        if actual_len < window_size:
            clarified_window = clarified_window[..., :actual_len]

        # Apply fade and accumulate
        window_fade = fade[:actual_len].view(1, 1, 1, -1)
        clarified[..., start:end] += clarified_window * window_fade
        weights[..., start:end] += window_fade

    # Normalize by weights
    clarified = clarified / weights.clamp(min=1e-8)

    if not had_batch:
        clarified = clarified.squeeze(0)

    return clarified


@torch.no_grad()
def decode_to_audio(
    dcae,
    latent: torch.Tensor,
    sr_out: int = 48000,
    device: str = "cuda",
) -> tuple:
    """
    Decode latent to audio using DCAE.

    Args:
        dcae: MusicDCAE model
        latent: [8, 16, T] or [B, 8, 16, T]
        sr_out: Output sample rate

    Returns:
        (sample_rate, audio_tensor)
    """
    if dcae is None:
        raise RuntimeError("DCAE not loaded. Cannot decode to audio.")

    # Ensure batch dimension
    had_batch = latent.dim() == 4
    if not had_batch:
        latent = latent.unsqueeze(0)

    latent = latent.to(device)
    B, C, H, T_slow = latent.shape

    # Calculate expected audio length
    audio_len = int(round(T_slow * DCAE_HOP * (sr_out / DCAE_SR)))
    audio_lengths = torch.tensor([audio_len] * B, device=device, dtype=torch.long)

    # Match DCAE dtype
    dcae_param = next(dcae.parameters())
    latent = latent.to(dtype=dcae_param.dtype)

    # Decode
    sr_pred, wav_pred = dcae.decode(latent, audio_lengths=audio_lengths, sr=sr_out)

    if not had_batch:
        wav_pred = wav_pred.squeeze(0)

    return sr_pred, wav_pred.cpu()


def process_single_latent(
    model: torch.nn.Module,
    dcae,
    input_path: str,
    output_dir: str,
    group_id: int,
    subgroup_id: int,
    device: str = "cuda",
    sr_out: int = 48000,
):
    """Process a single latent file."""
    os.makedirs(output_dir, exist_ok=True)

    # Load latent
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        latent = data
    elif isinstance(data, dict):
        latent = data.get("synthetic") or data.get("latent") or data.get("latents")
    else:
        raise ValueError(f"Unknown data format in {input_path}")

    if latent.dim() == 4:
        latent = latent.squeeze(0)

    print(f"Input latent shape: {latent.shape}")

    # Clarify
    clarified = clarify_latent(model, latent, group_id, subgroup_id, device=device)

    # Save clarified latent
    stem = Path(input_path).stem
    torch.save(clarified.cpu(), os.path.join(output_dir, f"{stem}_clarified.pt"))

    # Decode to audio if DCAE available
    if dcae is not None:
        # Decode input
        sr, wav_input = decode_to_audio(dcae, latent, sr_out=sr_out, device=device)
        torchaudio.save(
            os.path.join(output_dir, f"{stem}_input.wav"),
            wav_input.float(),
            sr
        )

        # Decode clarified
        sr, wav_clarified = decode_to_audio(dcae, clarified, sr_out=sr_out, device=device)
        torchaudio.save(
            os.path.join(output_dir, f"{stem}_clarified.wav"),
            wav_clarified.float(),
            sr
        )

        print(f"Saved audio to {output_dir}")


def process_pairs_comparison(
    model: torch.nn.Module,
    dcae,
    pairs_dir: str,
    output_dir: str,
    device: str = "cuda",
    sr_out: int = 48000,
    max_samples: Optional[int] = None,
):
    """
    Process pairs and generate comparison audio:
    - synthetic (input to clarifier)
    - clarified (output of clarifier)
    - real (ground truth target)
    """
    os.makedirs(output_dir, exist_ok=True)

    pairs_path = Path(pairs_dir)
    pair_files = sorted(pairs_path.glob("*.pt"))

    if max_samples:
        pair_files = pair_files[:max_samples]

    print(f"Processing {len(pair_files)} pairs...")

    for pair_file in tqdm(pair_files):
        try:
            data = torch.load(pair_file, map_location="cpu", weights_only=False)

            synthetic = data["synthetic"]
            real = data["real"]
            group_id = data.get("group_id", 0)
            subgroup_id = data.get("subgroup_id", 0)

            if isinstance(group_id, torch.Tensor):
                group_id = int(group_id.item())
            if isinstance(subgroup_id, torch.Tensor):
                subgroup_id = int(subgroup_id.item())

            # Handle batch dim
            if synthetic.dim() == 4:
                synthetic = synthetic.squeeze(0)
            if real.dim() == 4:
                real = real.squeeze(0)

            # Clarify
            clarified = clarify_latent(model, synthetic, group_id, subgroup_id, device=device)

            stem = pair_file.stem

            # Save latents
            torch.save({
                "synthetic": synthetic.cpu(),
                "clarified": clarified.cpu(),
                "real": real.cpu(),
                "group_id": group_id,
                "subgroup_id": subgroup_id,
            }, os.path.join(output_dir, f"{stem}_comparison.pt"))

            # Decode to audio if available
            if dcae is not None:
                # Synthetic
                sr, wav = decode_to_audio(dcae, synthetic, sr_out=sr_out, device=device)
                torchaudio.save(
                    os.path.join(output_dir, f"{stem}_1_synthetic.wav"),
                    wav.float(), sr
                )

                # Clarified
                sr, wav = decode_to_audio(dcae, clarified, sr_out=sr_out, device=device)
                torchaudio.save(
                    os.path.join(output_dir, f"{stem}_2_clarified.wav"),
                    wav.float(), sr
                )

                # Real
                sr, wav = decode_to_audio(dcae, real, sr_out=sr_out, device=device)
                torchaudio.save(
                    os.path.join(output_dir, f"{stem}_3_real.wav"),
                    wav.float(), sr
                )

        except Exception as e:
            print(f"Error processing {pair_file.name}: {e}")
            continue

    print(f"Done! Outputs saved to {output_dir}")


def compute_metrics(
    model: torch.nn.Module,
    pairs_dir: str,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics on test pairs.

    Returns dict with:
    - l1_before: L1 error between synthetic and real (before clarification)
    - l1_after: L1 error between clarified and real (after clarification)
    - improvement: relative improvement
    """
    pairs_path = Path(pairs_dir)
    pair_files = sorted(pairs_path.glob("*.pt"))

    if max_samples:
        pair_files = pair_files[:max_samples]

    l1_before_sum = 0.0
    l1_after_sum = 0.0
    count = 0

    for pair_file in tqdm(pair_files, desc="Computing metrics"):
        try:
            data = torch.load(pair_file, map_location="cpu", weights_only=False)

            synthetic = data["synthetic"]
            real = data["real"]
            group_id = data.get("group_id", 0)
            subgroup_id = data.get("subgroup_id", 0)

            if isinstance(group_id, torch.Tensor):
                group_id = int(group_id.item())
            if isinstance(subgroup_id, torch.Tensor):
                subgroup_id = int(subgroup_id.item())

            if synthetic.dim() == 4:
                synthetic = synthetic.squeeze(0)
            if real.dim() == 4:
                real = real.squeeze(0)

            # Ensure same length
            T_min = min(synthetic.shape[-1], real.shape[-1])
            synthetic = synthetic[..., :T_min]
            real = real[..., :T_min]

            # Clarify
            clarified = clarify_latent(model, synthetic, group_id, subgroup_id, device=device)

            # Compute L1
            l1_before = F.l1_loss(synthetic, real).item()
            l1_after = F.l1_loss(clarified.cpu(), real).item()

            l1_before_sum += l1_before
            l1_after_sum += l1_after
            count += 1

        except Exception as e:
            print(f"Error on {pair_file.name}: {e}")
            continue

    if count == 0:
        return {"l1_before": 0.0, "l1_after": 0.0, "improvement": 0.0}

    l1_before_avg = l1_before_sum / count
    l1_after_avg = l1_after_sum / count
    improvement = (l1_before_avg - l1_after_avg) / l1_before_avg * 100

    return {
        "l1_before": l1_before_avg,
        "l1_after": l1_after_avg,
        "improvement_pct": improvement,
        "num_samples": count,
    }


def main():
    parser = argparse.ArgumentParser(description="Test InstrumentClarifier")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to clarifier checkpoint")
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["base", "large"])
    parser.add_argument("--ace_checkpoint", type=str,
                        default="/home/arlo/Data/ACE-Step/checkpoints",
                        help="Path to ACE-Step checkpoints for DCAE")

    # Input modes (mutually exclusive)
    parser.add_argument("--input", type=str, default=None,
                        help="Single latent file to process")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of latent files to process")
    parser.add_argument("--pairs_dir", type=str, default=None,
                        help="Directory of pair files for comparison")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sr_out", type=int, default=48000,
                        help="Output audio sample rate")
    parser.add_argument("--max_samples", type=int, default=None)

    # For single file mode
    parser.add_argument("--group_id", type=int, default=4,
                        help="Instrument group ID (default: 4=brass)")
    parser.add_argument("--subgroup_id", type=int, default=7,
                        help="Instrument subgroup ID (default: 7=trumpet)")

    # Metrics mode
    parser.add_argument("--metrics_only", action="store_true",
                        help="Only compute metrics, don't decode audio")

    args = parser.parse_args()

    # Load model
    model = load_clarifier(args.checkpoint, args.model_size, args.device)

    # Load DCAE if not metrics-only
    dcae = None
    if not args.metrics_only:
        dcae = load_dcae(args.ace_checkpoint, args.device)

    # Process based on mode
    if args.metrics_only and args.pairs_dir:
        metrics = compute_metrics(model, args.pairs_dir, args.device, args.max_samples)
        print("\n=== Evaluation Metrics ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    elif args.input:
        process_single_latent(
            model, dcae, args.input, args.output_dir,
            args.group_id, args.subgroup_id,
            args.device, args.sr_out
        )

    elif args.pairs_dir:
        process_pairs_comparison(
            model, dcae, args.pairs_dir, args.output_dir,
            args.device, args.sr_out, args.max_samples
        )

    elif args.input_dir:
        # Process directory of single latents
        input_path = Path(args.input_dir)
        latent_files = sorted(input_path.glob("*.pt"))
        if args.max_samples:
            latent_files = latent_files[:args.max_samples]

        for lf in tqdm(latent_files, desc="Processing"):
            try:
                process_single_latent(
                    model, dcae, str(lf), args.output_dir,
                    args.group_id, args.subgroup_id,
                    args.device, args.sr_out
                )
            except Exception as e:
                print(f"Error on {lf.name}: {e}")

    else:
        parser.error("Must specify --input, --input_dir, or --pairs_dir")


if __name__ == "__main__":
    main()
