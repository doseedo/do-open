#!/usr/bin/env python3
"""
Preprocess audio pairs for AudioEnhancer training.

Creates (original_audio, decoded_dcae_audio) pairs and saves them to disk.
This avoids DCAE encode/decode on the fly during training.
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data/ACE-Step')

# Group/subgroup mappings
GROUP_TO_ID = {
    'strings': 0, 'woodwinds': 1, 'keys': 2, 'percussion': 3,
    'brass': 4, 'vocals': 5,
}

SUBGROUP_TO_ID = {
    'violin': 0, 'viola': 1, 'cello': 2, 'double_bass': 3,
    'flute': 4, 'oboe': 5, 'clarinet': 6, 'bassoon': 7,
    'french_horn': 8, 'piano': 9, 'organ': 10, 'harpsichord': 11,
    'drums': 12, 'trumpet': 13, 'trombone': 14, 'tuba': 15,
    'sax': 16, 'soprano': 17, 'alto': 18, 'tenor': 19, 'bass': 20,
}


def load_dcae(device='cuda'):
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
    base = '/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c'
    dcae = MusicDCAE(
        dcae_checkpoint_path=f'{base}/music_dcae_f8c8',
        vocoder_checkpoint_path=f'{base}/music_vocoder'
    )
    dcae.to(device).eval()
    return dcae


def load_audio(path: str, sample_rate: int = 48000) -> torch.Tensor:
    """Load audio and resample to target rate."""
    audio, sr = torchaudio.load(path)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    # Convert to stereo if mono
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return audio


def process_entry(
    entry: dict,
    dcae,
    output_dir: Path,
    segment_samples: int,
    sample_rate: int,
    device: str,
    idx: int,
) -> bool:
    """Process a single entry and save the pair."""
    try:
        audio_path = entry['audio_path']
        audio = load_audio(audio_path, sample_rate)

        T = audio.shape[-1]
        if T < segment_samples // 2:
            return False  # Too short

        # Random segment (or pad if short)
        if T <= segment_samples:
            pad = segment_samples - T
            audio = torch.nn.functional.pad(audio, (0, pad))
            start = 0
        else:
            start = random.randint(0, T - segment_samples)
            audio = audio[:, start:start + segment_samples]

        # Encode/decode through DCAE
        with torch.no_grad():
            audio_gpu = audio.unsqueeze(0).to(device)  # [1, 2, T]
            latent, _ = dcae.encode(audio_gpu)
            audio_len = torch.tensor([segment_samples], device=device)
            _, decoded = dcae.decode(latent, audio_lengths=audio_len, sr=sample_rate)

            if isinstance(decoded, list):
                decoded = decoded[0]
            decoded = decoded.squeeze(0).cpu()  # [2, T]

        # Match lengths
        min_len = min(audio.shape[-1], decoded.shape[-1])
        audio = audio[:, :min_len]
        decoded = decoded[:, :min_len]

        # Save pair
        pair_data = {
            'original': audio,  # [2, T]
            'decoded': decoded,  # [2, T]
            'group_id': entry['group_id'],
            'subgroup_id': entry['subgroup_id'],
            'group': entry['group'],
            'subgroup': entry['subgroup'],
            'source_path': audio_path,
            'start_sample': start,
        }

        torch.save(pair_data, output_dir / f'pair_{idx:06d}.pt')
        return True

    except Exception as e:
        print(f"Error processing {entry.get('audio_path', 'unknown')}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio enhancer pairs")
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--groups', type=str, nargs='+', default=None,
                        help='Filter by groups (e.g., brass strings)')
    parser.add_argument('--segment_seconds', type=float, default=3.0)
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--max_pairs', type=int, default=None,
                        help='Max pairs to generate (default: all)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Filter entries
    entries = []
    for entry in manifest:
        if entry.get('ensemble_detected') or entry.get('session_ensemble_flagged'):
            continue

        group = entry.get('group', '')
        subgroup = entry.get('sub_group', '')

        if args.groups and group not in args.groups:
            continue

        audio_path = entry.get('audio_path', '')
        if not audio_path or not os.path.exists(audio_path):
            continue

        if group not in GROUP_TO_ID:
            continue

        entries.append({
            'audio_path': audio_path,
            'group_id': GROUP_TO_ID[group],
            'subgroup_id': SUBGROUP_TO_ID.get(subgroup, 0),
            'group': group,
            'subgroup': subgroup,
        })

    print(f"Found {len(entries)} valid entries")

    # Count by subgroup
    by_subgroup = {}
    for e in entries:
        sg = e['subgroup']
        by_subgroup[sg] = by_subgroup.get(sg, 0) + 1
    for sg, count in sorted(by_subgroup.items()):
        print(f"  {sg}: {count}")

    if args.max_pairs:
        entries = entries[:args.max_pairs]
        print(f"Limited to {len(entries)} entries")

    # Load DCAE
    print("Loading DCAE...")
    dcae = load_dcae(args.device)
    print("DCAE loaded")

    segment_samples = int(args.segment_seconds * args.sample_rate)

    # Process entries
    success_count = 0
    for idx, entry in enumerate(tqdm(entries, desc="Processing")):
        if process_entry(entry, dcae, output_dir, segment_samples,
                        args.sample_rate, args.device, idx):
            success_count += 1

    print(f"\nDone! Created {success_count} pairs in {output_dir}")

    # Save config
    config = {
        'manifest': args.manifest,
        'groups': args.groups,
        'segment_seconds': args.segment_seconds,
        'sample_rate': args.sample_rate,
        'total_pairs': success_count,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
