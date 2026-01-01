#!/usr/bin/env python3
"""
Generate Synthetic Muted Data

Step 3: After validating the translator, generate synthetic muted audio
from the entire dry trumpet corpus.

This creates paired data: (dry_audio, synthetic_muted_audio) for training
the student model.

Usage:
    python generate_synthetic.py \
        --checkpoint ./checkpoints/best.pt \
        --output_dir /path/to/synthetic_data \
        --num_workers 4
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/mute_translator')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/dø')

from models import MuteTranslator, MuteTranslatorLarge
from dataset import load_manifest


# Global for worker processes
_dcae = None
_translator = None
_device = None


def init_worker(dcae_checkpoint: str, translator_checkpoint: str, device: str):
    """Initialize models in worker process."""
    global _dcae, _translator, _device

    _device = device

    # Load DCAE
    from do.pipeline_do import DoTrainComponents
    comps = DoTrainComponents(checkpoint_dir=dcae_checkpoint, dtype="float32")
    _dcae = comps.load_dcae()
    _dcae = _dcae.to(device).eval()

    # Load translator
    checkpoint = torch.load(translator_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if 'input_proj.weight' in state_dict:
        in_channels = state_dict['input_proj.weight'].shape[1]
        if in_channels == 8:
            _translator = MuteTranslator()
        else:
            _translator = MuteTranslatorLarge()
    else:
        _translator = MuteTranslator()

    _translator.load_state_dict(state_dict)
    _translator = _translator.to(device).eval()


def process_single_file(args: tuple) -> dict:
    """Process a single audio file."""
    audio_path, output_dir, save_latents = args

    global _dcae, _translator, _device

    result = {
        'input': audio_path,
        'success': False,
    }

    try:
        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        audio = audio.unsqueeze(0).to(_device)

        # Encode
        with torch.no_grad():
            dry_latent = _dcae.encode(audio, sr=sr)

            # Translate
            muted_latent = _translator(dry_latent)

            # Decode
            muted_audio = _dcae.decode(muted_latent)

        # Prepare output paths
        stem = Path(audio_path).stem
        parent_name = Path(audio_path).parent.name

        audio_output_dir = Path(output_dir) / "audio" / parent_name
        audio_output_dir.mkdir(parents=True, exist_ok=True)

        # Save audio
        muted_audio = muted_audio.squeeze(0).cpu()
        muted_audio = muted_audio / (muted_audio.abs().max() + 1e-8) * 0.9

        output_path = audio_output_dir / f"{stem}_muted.wav"
        torchaudio.save(str(output_path), muted_audio, 44100)

        result['output_audio'] = str(output_path)

        # Optionally save latents
        if save_latents:
            latent_output_dir = Path(output_dir) / "latents" / parent_name
            latent_output_dir.mkdir(parents=True, exist_ok=True)

            dry_latent_path = latent_output_dir / f"{stem}_dry.pt"
            muted_latent_path = latent_output_dir / f"{stem}_muted.pt"

            torch.save(dry_latent.cpu(), str(dry_latent_path))
            torch.save(muted_latent.cpu(), str(muted_latent_path))

            result['dry_latent'] = str(dry_latent_path)
            result['muted_latent'] = str(muted_latent_path)

        result['success'] = True
        result['duration_sec'] = audio.shape[-1] / sr

    except Exception as e:
        result['error'] = str(e)

    return result


class SyntheticDataGenerator:
    """Generate synthetic muted data from dry trumpet corpus."""

    def __init__(
        self,
        translator_checkpoint: str,
        dcae_checkpoint: str,
        manifest_path: str,
        output_dir: str,
        device: str = "cuda",
        num_workers: int = 1,
        save_latents: bool = True,
        max_files: int = None,
        skip_existing: bool = True,
    ):
        self.translator_checkpoint = translator_checkpoint
        self.dcae_checkpoint = dcae_checkpoint
        self.manifest_path = manifest_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.num_workers = num_workers
        self.save_latents = save_latents
        self.max_files = max_files
        self.skip_existing = skip_existing

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load manifest
        self.dry_entries, self.muted_entries = load_manifest(manifest_path)
        print(f"Found {len(self.dry_entries)} dry trumpet files")

        # Filter to existing files
        self.dry_entries = [
            e for e in self.dry_entries
            if os.path.exists(e.get('audio_path', ''))
        ]
        print(f"Accessible files: {len(self.dry_entries)}")

        if max_files:
            self.dry_entries = self.dry_entries[:max_files]
            print(f"Limited to: {len(self.dry_entries)} files")

    def generate_sequential(self):
        """Generate sequentially (single GPU)."""
        print("\nInitializing models...")
        init_worker(self.dcae_checkpoint, self.translator_checkpoint, self.device)

        results = []
        total_duration = 0.0

        pbar = tqdm(self.dry_entries, desc="Generating synthetic muted")
        for entry in pbar:
            audio_path = entry['audio_path']

            # Check if already exists
            if self.skip_existing:
                stem = Path(audio_path).stem
                parent = Path(audio_path).parent.name
                expected_output = self.output_dir / "audio" / parent / f"{stem}_muted.wav"
                if expected_output.exists():
                    continue

            result = process_single_file((audio_path, str(self.output_dir), self.save_latents))
            results.append(result)

            if result['success']:
                total_duration += result.get('duration_sec', 0)

            pbar.set_postfix({
                'success': sum(1 for r in results if r['success']),
                'duration': f"{total_duration/3600:.1f}h"
            })

        return results

    def generate(self):
        """Generate synthetic muted data."""
        print("\n" + "=" * 60)
        print("SYNTHETIC MUTED DATA GENERATION")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Files to process: {len(self.dry_entries)}")
        print(f"Save latents: {self.save_latents}")
        print(f"Device: {self.device}")

        start_time = datetime.now()

        # Use sequential for now (GPU memory management)
        results = self.generate_sequential()

        # Summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]

        total_duration = sum(r.get('duration_sec', 0) for r in successful)

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total audio duration: {total_duration/3600:.2f} hours")
        print(f"Processing time: {datetime.now() - start_time}")
        print(f"Output: {self.output_dir}")

        # Save manifest
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'translator_checkpoint': self.translator_checkpoint,
            'num_files': len(successful),
            'total_duration_hours': total_duration / 3600,
            'files': [
                {
                    'dry_audio': r['input'],
                    'muted_audio': r['output_audio'],
                    'dry_latent': r.get('dry_latent'),
                    'muted_latent': r.get('muted_latent'),
                }
                for r in successful
            ]
        }

        manifest_path = self.output_dir / "synthetic_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to: {manifest_path}")

        # Save failed list
        if failed:
            failed_path = self.output_dir / "failed_files.json"
            with open(failed_path, 'w') as f:
                json.dump(failed, f, indent=2)
            print(f"Failed files list: {failed_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Muted Data")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to translator checkpoint')
    parser.add_argument('--dcae_checkpoint', type=str,
                        default='/home/arlo/Data/ACE-Step/checkpoints',
                        help='Path to DCAE checkpoint directory')
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/Data.backup/final_training_manifest_brass_only.json',
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for synthetic data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (1 recommended for GPU)')
    parser.add_argument('--save_latents', action='store_true',
                        help='Also save latent tensors')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit number of files to process')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip files that already have output')

    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        translator_checkpoint=args.checkpoint,
        dcae_checkpoint=args.dcae_checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
        save_latents=args.save_latents,
        max_files=args.max_files,
        skip_existing=args.skip_existing,
    )

    generator.generate()


if __name__ == "__main__":
    main()
