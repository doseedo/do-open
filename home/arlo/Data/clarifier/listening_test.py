#!/usr/bin/env python3
"""
Listening test script for InstrumentClarifier.
Generates audio comparisons: ground truth vs synthetic vs clarified.
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np

sys.path.insert(0, '/home/arlo/Data/clarifier')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from models import InstrumentClarifier


def load_dcae(device='cuda'):
    """Load MusicDCAE for decoding latents to audio."""
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    base = '/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c'
    dcae = MusicDCAE(
        dcae_checkpoint_path=f'{base}/music_dcae_f8c8',
        vocoder_checkpoint_path=f'{base}/music_vocoder'
    )
    dcae.to(device)
    return dcae


def decode_latent(dcae, latent, device='cuda'):
    """Decode DCAE latent to audio tensor."""
    with torch.no_grad():
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        latent = latent.to(device).float()
        T = latent.shape[-1]
        audio_len = T * 512
        audio_lengths = torch.tensor([audio_len], device=device)
        sr, audio = dcae.decode(latent, audio_lengths=audio_lengths, sr=48000)
        if isinstance(audio, list):
            audio = audio[0]
        audio = audio.squeeze()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return audio.cpu()


def run_listening_test(
    clarifier_ckpt: str,
    pairs_dir: str,
    output_dir: str,
    samples_per_instrument: int = 3,
    instruments: list = None,
    group_vocab: int = 6,
    subgroup_vocab: int = 20,
    device: str = 'cuda',
    seed: int = 42,
):
    """
    Run listening test.

    Args:
        clarifier_ckpt: Path to clarifier checkpoint
        pairs_dir: Directory with pair .pt files
        output_dir: Where to save audio files
        samples_per_instrument: Number of samples per instrument
        instruments: List of instruments to test (None = all)
        group_vocab: Group vocabulary size
        subgroup_vocab: Subgroup vocabulary size
        device: cuda or cpu
        seed: Random seed for sample selection
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load clarifier
    print(f"Loading clarifier from {clarifier_ckpt}")
    model = InstrumentClarifier(group_vocab=group_vocab, subgroup_vocab=subgroup_vocab).to(device)
    ckpt = torch.load(clarifier_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Best val loss: {ckpt.get('best_val_loss', 'N/A'):.4f}")

    # Load DCAE
    print("Loading DCAE...")
    dcae = load_dcae(device)
    print("  Done")

    # Group pair files by instrument
    print(f"Scanning pairs in {pairs_dir}")
    pairs_by_instrument = {}
    for pf in Path(pairs_dir).glob('pair_*.pt'):
        data = torch.load(pf, map_location='cpu')
        meta = data.get('meta', {})
        inst = meta.get('subgroup', 'unknown')
        if inst not in pairs_by_instrument:
            pairs_by_instrument[inst] = []
        pairs_by_instrument[inst].append(str(pf))

    print(f"Found instruments: {list(pairs_by_instrument.keys())}")

    # Filter instruments if specified
    if instruments:
        pairs_by_instrument = {k: v for k, v in pairs_by_instrument.items() if k in instruments}

    # Run test
    all_mse_before, all_mse_after = [], []
    all_cossim_before, all_cossim_after = [], []

    idx = 0
    for instrument, pair_files in sorted(pairs_by_instrument.items()):
        print(f"\n{instrument} ({len(pair_files)} pairs):")

        # Sample files
        selected = random.sample(pair_files, min(samples_per_instrument, len(pair_files)))

        for i, pf in enumerate(selected):
            data = torch.load(pf, map_location='cpu')
            synthetic = data['synthetic'].unsqueeze(0).to(device).float()
            real = data['real'].unsqueeze(0).to(device).float()
            group_id = torch.tensor([data['group_id']], device=device)
            subgroup_id = torch.tensor([data['subgroup_id']], device=device)

            with torch.no_grad():
                clarified = model(synthetic, group_id, subgroup_id)

            # Metrics
            mse_before = ((synthetic - real) ** 2).mean().item()
            mse_after = ((clarified - real) ** 2).mean().item()
            cossim_before = torch.nn.functional.cosine_similarity(
                synthetic.view(-1), real.view(-1), dim=0
            ).item()
            cossim_after = torch.nn.functional.cosine_similarity(
                clarified.view(-1), real.view(-1), dim=0
            ).item()

            all_mse_before.append(mse_before)
            all_mse_after.append(mse_after)
            all_cossim_before.append(cossim_before)
            all_cossim_after.append(cossim_after)

            mse_change = (mse_after - mse_before) / mse_before * 100
            print(f"  {i+1}: MSE {mse_before:.4f}->{mse_after:.4f} ({mse_change:+.1f}%), "
                  f"CosSim {cossim_before:.4f}->{cossim_after:.4f}")

            # Decode and save audio
            gt_audio = decode_latent(dcae, real[0], device)
            syn_audio = decode_latent(dcae, synthetic[0], device)
            clar_audio = decode_latent(dcae, clarified[0], device)

            prefix = f"{idx:02d}_{instrument}_{i+1}"
            torchaudio.save(f"{output_dir}/{prefix}_gt.wav", gt_audio, 48000)
            torchaudio.save(f"{output_dir}/{prefix}_synthetic.wav", syn_audio, 48000)
            torchaudio.save(f"{output_dir}/{prefix}_clarified.wav", clar_audio, 48000)
            idx += 1

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    avg_mse_before = np.mean(all_mse_before)
    avg_mse_after = np.mean(all_mse_after)
    mse_reduction = (avg_mse_after - avg_mse_before) / avg_mse_before * 100

    avg_cossim_before = np.mean(all_cossim_before)
    avg_cossim_after = np.mean(all_cossim_after)
    cossim_improvement = avg_cossim_after - avg_cossim_before

    print(f"MSE:    {avg_mse_before:.4f} -> {avg_mse_after:.4f} ({mse_reduction:+.1f}%)")
    print(f"CosSim: {avg_cossim_before:.4f} -> {avg_cossim_after:.4f} ({cossim_improvement:+.4f})")
    print(f"\nGenerated {idx * 3} audio files in {output_dir}/")

    # Save summary
    summary = {
        'checkpoint': clarifier_ckpt,
        'pairs_dir': pairs_dir,
        'samples_per_instrument': samples_per_instrument,
        'total_samples': idx,
        'mse_before': avg_mse_before,
        'mse_after': avg_mse_after,
        'mse_reduction_pct': mse_reduction,
        'cossim_before': avg_cossim_before,
        'cossim_after': avg_cossim_after,
        'cossim_improvement': cossim_improvement,
    }
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Clarifier listening test")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to clarifier checkpoint')
    parser.add_argument('--pairs_dir', type=str, required=True,
                        help='Directory with pair .pt files')
    parser.add_argument('--output_dir', type=str, default='/tmp/clarifier_listening_test',
                        help='Output directory for audio files')
    parser.add_argument('--samples', type=int, default=3,
                        help='Samples per instrument')
    parser.add_argument('--instruments', type=str, nargs='+', default=None,
                        help='Specific instruments to test (default: all)')
    parser.add_argument('--group_vocab', type=int, default=6)
    parser.add_argument('--subgroup_vocab', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    run_listening_test(
        clarifier_ckpt=args.checkpoint,
        pairs_dir=args.pairs_dir,
        output_dir=args.output_dir,
        samples_per_instrument=args.samples,
        instruments=args.instruments,
        group_vocab=args.group_vocab,
        subgroup_vocab=args.subgroup_vocab,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
