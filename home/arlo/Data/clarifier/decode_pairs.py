#!/usr/bin/env python3
"""
Decode clarifier pairs to audio for listening tests.

Usage:
    python decode_pairs.py --pairs_dir /mnt/msdd2/clarifier_pairs_brass --output_dir /tmp/listening_test
    python decode_pairs.py --pairs_dir /mnt/msdd2/clarifier_pairs_brass --output_dir /tmp/listening_test --max 5
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/dø')


def load_dcae(checkpoint_dir='/home/arlo/Data/ACE-Step/checkpoints'):
    """Load DCAE model."""
    from do.pipeline_do import DoTrainComponents
    comps = DoTrainComponents(checkpoint_dir=checkpoint_dir, dtype="float32")
    dcae = comps.load_dcae()
    dcae = dcae.cuda().eval()
    return dcae


def decode_latent(dcae, latent):
    """Decode latent to audio. Returns [2, samples] tensor at 44100 Hz."""
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)

    with torch.no_grad():
        sr, audio_list = dcae.decode(latent.cuda())

    audio = audio_list[0].cpu()  # [2, samples]
    return audio, sr


def decode_pairs(pairs_dir, output_dir, max_pairs=None):
    """Decode pairs to audio files."""
    pairs_dir = Path(pairs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find pair files
    pair_files = sorted(pairs_dir.glob("*.pt"))
    if max_pairs:
        pair_files = pair_files[:max_pairs]

    print(f"Found {len(pair_files)} pairs to decode")

    # Load DCAE
    print("Loading DCAE...")
    dcae = load_dcae()

    print(f"\nDecoding to {output_dir}...\n")

    for pf in pair_files:
        data = torch.load(pf, map_location="cpu", weights_only=False)
        syn = data['synthetic']
        real = data['real']
        meta = data.get('meta', {})
        subgroup = meta.get('subgroup', 'unknown')

        # Decode
        syn_audio, sr = decode_latent(dcae, syn)
        real_audio, _ = decode_latent(dcae, real)

        # Normalize
        syn_audio = syn_audio / (syn_audio.abs().max() + 1e-8) * 0.9
        real_audio = real_audio / (real_audio.abs().max() + 1e-8) * 0.9

        # Save
        base = pf.stem
        syn_path = output_dir / f"{base}_{subgroup}_synthetic.wav"
        real_path = output_dir / f"{base}_{subgroup}_real.wav"

        torchaudio.save(str(syn_path), syn_audio, sr)
        torchaudio.save(str(real_path), real_audio, sr)

        duration = real_audio.shape[-1] / sr
        print(f"{pf.name} ({subgroup}): {duration:.1f}s")

    print(f"\n✓ Done! Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Decode clarifier pairs to audio")
    parser.add_argument("--pairs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max", type=int, default=None, help="Max pairs to decode")

    args = parser.parse_args()
    decode_pairs(args.pairs_dir, args.output_dir, args.max)


if __name__ == "__main__":
    main()
