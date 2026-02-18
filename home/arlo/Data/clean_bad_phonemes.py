#!/usr/bin/env python3
"""Remove bad phoneme data from all tensor files so they can be reprocessed"""
import torch
import json
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/home/arlo/Data/vocal_training_manifest_READY_NOGROUPS_SPKEMB.json')
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))

    print(f"Removing bad phoneme data from {len(manifest)} files...")

    removed_count = 0
    error_count = 0

    for item in tqdm(manifest, desc="Cleaning phoneme data"):
        vc_paths = item.get('vocal_conditioning_paths', {})
        tensor_path = vc_paths.get('lyrics_tensors')

        if tensor_path and Path(tensor_path).exists():
            try:
                tensors = torch.load(tensor_path, map_location='cpu')

                # Check if has phoneme data
                if 'phoneme_frames' in tensors:
                    # Remove phoneme keys
                    keys_to_remove = ['phoneme_frames', 'phoneme_boundaries', 'phoneme_confidence',
                                     'phoneme_timings', 'num_phonemes', 'phoneme_vocab_size']
                    for key in keys_to_remove:
                        if key in tensors:
                            del tensors[key]

                    # Save back
                    torch.save(tensors, tensor_path)
                    removed_count += 1
            except Exception as e:
                error_count += 1

    print(f"\n{'='*60}")
    print(f"✅ Removed phoneme data from {removed_count} files")
    print(f"❌ Errors: {error_count}")
    print(f"{'='*60}")
    print(f"\nNow run:")
    print(f"  python3 add_phoneme_targets.py --process_all --workers 8 --manifest {args.manifest}")

if __name__ == '__main__':
    main()
